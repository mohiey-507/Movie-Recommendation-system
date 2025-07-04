import torch
import pandas as pd 
from torch.utils.data import Dataset

class CollaborativeDataset(Dataset):
    def __init__(self, ratings_df: pd.DataFrame):
        self.users = torch.tensor(ratings_df['userId'].values, dtype=torch.long)
        self.movies = torch.tensor(ratings_df['movieId'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            'user': self.users[idx],
            'movie': self.movies[idx],
            'rating': self.ratings[idx]
        }


class ContentDataset(Dataset):
    def __init__(self, df, u2i, m2i, genres_mat, user_genres_mat, movie_avg, user_avg):
        self.user_idx  = torch.tensor(df["userId"].map(u2i).values, dtype=torch.long)
        self.movie_idx = torch.tensor(df["movieId"].map(m2i).values, dtype=torch.long)
        self.rating    = torch.tensor(df["rating"].values, dtype=torch.float32)

        self.genres_mat = genres_mat
        self.user_genres_mat  = user_genres_mat
        self.movie_avg  = movie_avg
        self.user_avg   = user_avg

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        m_idx = self.movie_idx[idx]
        u_idx = self.user_idx[idx]
        return {
            "user":      u_idx,
            "movie":     m_idx,
            "genres":    self.genres_mat[m_idx],
            "user_fav":  self.user_genres_mat[u_idx],
            "movie_avg": self.movie_avg[m_idx],
            "user_avg":  self.user_avg[u_idx],
            "rating":    self.rating[idx],
        }


class SequenceDataset(Dataset):
    def __init__(self, df, max_seq_len: int):
        self.hist = df['hist'].tolist()
        self.target = df['target'].tolist()
        self.rating = df['rating'].tolist()
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        hist = self.hist[idx]
        L = len(hist)
        pad = self.max_seq_len - L
        hist_pad = [0] * pad + [int(m) for m in hist]
        mask = [0] * pad + [1] * L
        return {
            'hist': torch.tensor(hist_pad, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'target': torch.tensor(int(self.target[idx]), dtype=torch.long),
            'rating': torch.tensor(float(self.rating[idx]), dtype=torch.float32)
        }
