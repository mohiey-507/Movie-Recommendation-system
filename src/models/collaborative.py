import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self,
                num_users: int,
                num_movies: int,
                movie_avg_tensor: torch.Tensor,
                emb_dim: int = 16,
                mlp_hidden: tuple[int, ...] = (64, 32, 16),
                dropout: float = 0.2
        ):
        super().__init__()

        self.register_buffer('movie_avg', movie_avg_tensor)

        self.user_bias  = nn.Embedding(num_users + 1, 1)
        self.item_bias  = nn.Embedding(num_movies + 1, 1)

        self.gmf_user_emb  = nn.Embedding(num_users + 1, emb_dim)
        self.gmf_item_emb  = nn.Embedding(num_movies + 1, emb_dim)

        self.mlp_user_emb  = nn.Embedding(num_users + 1, emb_dim)
        self.mlp_item_emb  = nn.Embedding(num_movies + 1, emb_dim)

        layers: list[nn.Module] = []
        in_dim = emb_dim * 2
        for h in mlp_hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h
        self.mlp = nn.Sequential(*layers)

        self.predict = nn.Linear(emb_dim + mlp_hidden[-1] + 2, 1)

    def forward(self, user_indices, item_indices):

        avg = self.movie_avg[item_indices]

        u_bias = self.user_bias(user_indices) # (B, 1)
        i_bias = self.item_bias(item_indices) # (B, 1)

        gmf_user = self.gmf_user_emb(user_indices) # (B, emb_dim)
        gmf_item = self.gmf_item_emb(item_indices) # (B, emb_dim)
        gmf_out = gmf_user * gmf_item

        mlp_user = self.mlp_user_emb(user_indices) # (B, mlp_layers[-1))
        mlp_item = self.mlp_item_emb(item_indices) # (B, mlp_layers[-1))

        mlp_in = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_out = self.mlp(mlp_in)

        x = torch.cat([gmf_out, mlp_out, u_bias, i_bias], dim=1)
        pred_residual = self.predict(x).squeeze(1)

        return pred_residual + avg
