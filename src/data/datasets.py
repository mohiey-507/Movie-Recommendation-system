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
    def __init__(self, df, u2i, m2i, genres_mat, tags_mat, movie_avg, user_avg):
        self.user_idx  = torch.tensor(df["userId"].map(u2i).values, dtype=torch.long)
        self.movie_idx = torch.tensor(df["movieId"].map(m2i).values, dtype=torch.long)
        self.rating    = torch.tensor(df["rating"].values, dtype=torch.float32)
        self.genres_mat = genres_mat
        self.tags_mat   = tags_mat
        self.movie_avg  = movie_avg
        self.user_avg   = user_avg

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        m_idx = self.movie_idx[idx]
        return {
            'user':      self.user_idx[idx],
            'genres':    self.genres_mat[m_idx],
            'tags':      self.tags_mat[m_idx],
            'movie_avg': self.movie_avg[m_idx],
            'user_avg':  self.user_avg[self.user_idx[idx]],
            'rating':    self.rating[idx]
        }