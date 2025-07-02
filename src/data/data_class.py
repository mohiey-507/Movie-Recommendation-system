from typing import Optional
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
    train_ds: Dataset
    val_ds  : Dataset

    user2idx : dict
    movie2idx: dict
    genre2idx: dict

    genres_mat     : torch.Tensor
    user_genres_mat: torch.Tensor

    movie_avg: torch.Tensor
    user_avg : torch.Tensor

    num_users : int
    num_movies: int
    num_genres: int

    movies_df : pd.DataFrame
    ratings_df: pd.DataFrame


@dataclass
class SeqContentData:
    train_ds : Dataset
    val_ds   : Dataset

    user2idx : dict
    movie2idx: dict
    genre2idx: dict
    year2idx : dict

    genres_mat   : torch.Tensor
    year_arr     : torch.Tensor
    numeric_feats: torch.Tensor

    num_users : int
    num_movies: int
    num_genres: int
    num_years : int

    max_seq_len : int

    overview_feats: Optional[torch.Tensor]
    overview_dim  : Optional[int]
