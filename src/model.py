import torch
import torch.nn as nn
import numpy

class DebaterEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.GRU(input_size=7, # num of expected features
                            hidden_size=16,
                            num_layers=1,
                            bias=True,
                            batch_first=True, # means we should batch the input (unsqueeze(0) ? )
                            )
        # note: dropout meaningless bc this is single layer (dropout is between layers)

    def forward(self, x, mask):
        # x shape (B, max history, token dim) mask shape (B, max history) real tokens at the front
        # returns (B, hidden dim)

        B = x.shape[0]
        out = x.new_zeros(B, self.layer.hidden_size)

        lengths = mask.sum(dim=1).long()
        nonempty = lengths > 0 # skip length 0
        if nonempty.any():
            packed = nn.utils.rnn.pack_padded_sequence(
                x[nonempty], lengths[nonempty].cpu(), # must be int64 on cpu
                batch_first=True, enforce_sorted=False,
            )
            _, hidden_state = self.layer(packed) # hidden_state: (num_layers=1, B_nonempty, hidden dim)
            out[nonempty] = hidden_state.squeeze(0)
        return out

class DebateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DebaterEncoder() # we use one encoder for each debater

        H = self.encoder.layer.hidden_size # 16
        repr_dim = H + 5
        
        self.strength_fn = nn.Sequential(
            nn.Linear(repr_dim, H),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(H, 1),
        )

        self.judge_term = nn.Linear(4, 1) # judge features turn into a scalar logit shift
        self.side_bias = nn.Parameter(torch.zeros(1)) # learned aff side advantage

        self.speaker_head = nn.Linear(repr_dim, 1) # dropped at inference

    def forward(self, x):
        # note that x is batched
        # x is a dictionary the dataloader yields
        static = x["static"] # shape (batch size, 10)
        aff_repr = self._team_repr(x["aff_seq"], x["aff_mask"], static[:, :5])
        neg_repr = self._team_repr(x["neg_seq"], x["neg_mask"], static[:, 5:])

        s_aff = self.strength_fn(aff_repr).squeeze(-1) # (Batch size)
        s_neg = self.strength_fn(neg_repr).squeeze(-1) # (Batch size)
        judge = self.judge_term(x["judge"]).squeeze(-1)  # (Batch size)
        logit = s_aff - s_neg + self.side_bias + judge  # (Batch size)

        speaker_preds = torch.cat([self.speaker_head(aff_repr), self.speaker_head(neg_repr)], dim=1) 
        
        return logit, speaker_preds

    def _team_repr(self, seq, mask, static_side):
        # seq shape (batch size, 2, 30, 7)
        # mask shape (batch size, 2, 30)
        # static_side shape (batch size, 5)
        # team_repr shape (B, hidden dim + 5)

        B = seq.shape[0]

        # encode both debaters in one batched call - fold 2 debaters into batch dim
        states = self.encoder(
            seq.reshape(-1, seq.shape[-2], seq.shape[-1]), # (batch dim * 2, 30, 7)
            mask.reshape(-1, mask.shape[-1]), # (batch dim * 2, 30)
        ) # gives (batch dim * 2, hidden dim)
        states = states.reshape(B, 2, -1) # (batch dim, 2, hidden dim)
        pooled = states.mean(dim=1) # (batch, hidden dim) mean over 2 debaters
        return torch.cat([pooled, static_side], dim=1) # (batch, hidden + 5)


