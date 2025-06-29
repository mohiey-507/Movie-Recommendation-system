from pathlib import Path

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ConfigLoader
from src import get_device
from .datasets import CollaborativeDataset

class CollaborativeProcessor:
    """
    Loads and preprocesses collaborative filtering data, splitting into train/val sets.

    Config keys required:
    - dirs: {'raw_data_dir': str}
    - collaborative: {'test_split_ratio': float}
    - random_seed: int
    """
    def __init__(self, cfg: dict = None, device: torch.device = None):
        # Defer loading and device resolution
        self.cfg = cfg or ConfigLoader.load_config()
        self.device = device or get_device()

        # Configuration sections
        self.dirs_cfg = self.cfg['dirs']
        self.collab_cfg = self.cfg['collaborative']
        self.test_size = self.collab_cfg['test_split_ratio']
        self.random_seed = self.cfg['random_seed']

        # Paths
        self.data_dir = Path(self.dirs_cfg['raw_data_dir'])

        # Internal state
        self._data_loaded = False

    def _load_and_normalize(self) -> None:
        # Load raw CSVs
        ratings_path = self.data_dir / 'ratings.csv'
        movies_path = self.data_dir / 'movies.csv'
        if not ratings_path.exists() or not movies_path.exists():
            raise FileNotFoundError(f"Data files not found in {self.data_dir}")

        ratings_df = pd.read_csv(ratings_path).drop(columns=['timestamp'])
        movies_df = pd.read_csv(movies_path).drop(columns=['genres'])

        # Compute statistics
        self.num_users = int(ratings_df['userId'].nunique())
        self.num_movies = int(ratings_df['movieId'].nunique())
        movie_means = ratings_df.groupby('movieId')['rating'].mean()
        global_mean = movie_means.mean()

        # Normalize ratings
        ratings_df['movie_avg'] = ratings_df['movieId'].map(movie_means).fillna(global_mean)
        ratings_df['rating'] = ratings_df['rating'] - ratings_df['movie_avg']

        means_array = movie_means.reindex(
            index=range(self.num_movies + 1),
            fill_value=global_mean
        ).to_numpy(dtype='float32')
        self.movie_avg_tensor = torch.tensor(means_array, device=self.device)

        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self._data_loaded = True

    def _split(self) -> None:
        self.train_df, self.val_df = train_test_split(
            self.ratings_df,
            test_size=self.test_size,
            random_state=self.random_seed
        )

    def prepare_data(self):
        """
        Returns:
        train_dataset, val_dataset, movie_avg_tensor, num_users, num_movies, movies_df, ratings_df
        """
        if not self._data_loaded:
            self._load_and_normalize()
            self._split()

        train_ds = CollaborativeDataset(self.train_df)
        val_ds = CollaborativeDataset(self.val_df)
        return (
            train_ds, val_ds, 
            self.movie_avg_tensor,
            self.num_users, self.num_movies,
            self.movies_df, self.ratings_df,
        )
