from pathlib import Path
from dataclasses import dataclass

import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src import get_device
from .datasets import CollaborativeDataset

@dataclass
class CollabData:
    train_ds: Dataset
    val_ds: Dataset
    movie_avg: torch.Tensor
    num_users: int
    num_movies: int
    movies_df: pd.DataFrame
    ratings_df: pd.DataFrame

def load_collab_data(
    data_dir: str,
    test_size: float,
    seed: int,
    device: torch.device = None
) -> CollabData:
    device = device or get_device()
    data_dir = Path(data_dir)

    ratings = pd.read_csv(data_dir / "ratings.csv").drop(columns=["timestamp"])
    movies  = pd.read_csv(data_dir / "movies.csv").drop(columns=["genres"])

    # Mean Normalization
    movie_means     = ratings.groupby("movieId")["rating"].mean()
    ratings["movie_avg"] = ratings["movieId"].map(movie_means)
    ratings["rating"]    = ratings["rating"] - ratings["movie_avg"]

    num_users  = int(ratings["userId"].nunique())
    num_movies = int(ratings["movieId"].nunique())

    # Per movie means tensor
    means_array = movie_means.reindex(
        range(num_movies + 1)
    ).to_numpy(dtype="float32")
    movie_avg = torch.tensor(means_array, device=device)

    train_df, val_df = train_test_split(
        ratings, test_size=test_size, random_state=seed
    )
    train_ds = CollaborativeDataset(train_df)
    val_ds   = CollaborativeDataset(val_df)

    return CollabData(
        train_ds=train_ds,
        val_ds=val_ds,
        movie_avg=movie_avg,
        num_users=num_users,
        num_movies=num_movies,
        movies_df=movies,
        ratings_df=ratings,
    )
