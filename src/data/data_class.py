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
