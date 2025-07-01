import torch
import torch.nn as nn


class ContentModel(nn.Module):
    def __init__(self, num_users, num_genres, emb_dim=32, hidden_dim=64):
        super().__init__()
        self.user_emb  = nn.Embedding(num_users, emb_dim)
        self.genre_emb = nn.Embedding(num_genres, emb_dim, padding_idx=0)
        self.num_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, emb_dim // 4)
        )

        # Fuse all: user + movie‑genre + movie‑tag + user‑fav‑genre + numeric
        total_dim = emb_dim * 3 + emb_dim // 4
        self.fuse = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user, genres, user_fav, movie_avg, user_avg):
        ue  = self.user_emb(user)                     # (B, emb)
        ge  = self.genre_emb(genres).mean(dim=1)      # (B, emb)
        uf  = self.genre_emb(user_fav).mean(dim=1)    # (B, emb)

        num = torch.stack([movie_avg, user_avg], dim=1)
        ne  = self.num_net(num)                       # (B, emb // 4)

        x   = torch.cat([ue, ge, uf, ne], dim=1)
        return self.fuse(x).squeeze(1)