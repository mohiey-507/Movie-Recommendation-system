from dataclasses import dataclass

import torch
import pandas as pd
from torch.utils.data import Dataset

@dataclass
class CollabData:
    train_ds: Dataset
    val_ds  : Dataset

    movie_avg: torch.Tensor

    num_users : int
    num_movies: int

    movies_df : pd.DataFrame
    ratings_df: pd.DataFrame

    def to(self, device: torch.device):
        self.movie_avg = self.movie_avg.to(device)
        for ds in [self.train_ds, self.val_ds]:
            ds.users = ds.users.to(device)
            ds.movies = ds.movies.to(device)
            ds.ratings = ds.ratings.to(device)
        return self

@dataclass
class ContentData:
    train_ds:            Dataset
    val_ds:              Dataset
    user2idx:            dict
    movie2idx:           dict
    genre2idx:           dict

    genres_mat:          torch.Tensor   # (num_movies, max_g=4)
    user_genres_mat:     torch.Tensor   # (num_users, user_fav_k=5)

    movie_avg:           torch.Tensor   # (num_movies,)
    user_avg:            torch.Tensor   # (num_users,)

    num_users:           int
    num_movies:          int
    num_genres:          int

    movies_df:           pd.DataFrame
    ratings_df:          pd.DataFrame

    def to(self, device: torch.device):
        self.genres_mat = self.genres_mat.to(device)
        self.user_genres_mat = self.user_genres_mat.to(device)
        self.movie_avg = self.movie_avg.to(device)
        self.user_avg = self.user_avg.to(device)
        for ds in [self.train_ds, self.val_ds]:
            ds.user_idx = ds.user_idx.to(device)
            ds.movie_idx = ds.movie_idx.to(device)
            ds.rating = ds.rating.to(device)
            ds.genres_mat = ds.genres_mat.to(device)
            ds.user_genres_mat = ds.user_genres_mat.to(device)
            ds.movie_avg = ds.movie_avg.to(device)
            ds.user_avg = ds.user_avg.to(device)
        return self