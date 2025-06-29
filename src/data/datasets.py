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
