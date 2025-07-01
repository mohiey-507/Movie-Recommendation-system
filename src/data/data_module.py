from pathlib import Path

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from src import get_device
from .datasets import (
    CollaborativeDataset, ContentDataset
)
from .data_class import (
    CollabData, ContentData
)


def load_collab_data(
    data_dir: str,
    test_size: float,
    seed: int,
    device: torch.device = None
) -> CollabData:
    device = device or get_device()
    data_dir = Path(data_dir)
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_file = processed_dir / f"collab_data_{test_size}_{seed}.pt"

    if processed_file.exists():
        data = torch.load(processed_file)
        return data.to(device)

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

    data = CollabData(
        train_ds=train_ds,
        val_ds=val_ds,
        movie_avg=movie_avg,
        num_users=num_users,
        num_movies=num_movies,
        movies_df=movies,
        ratings_df=ratings,
    )

    torch.save(data, processed_file)

    return data.to(device)


def load_content_data(
    data_dir: str,
    test_size: float,
    seed: int,
    max_g: int = 5,
    user_fav_k: int = 7,
    device: torch.device = None
) -> ContentData:
    device = device or get_device()
    data_dir = Path(data_dir)
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_file = processed_dir / f"content_data_{test_size}_{seed}_{max_g}_{user_fav_k}.pt"

    if processed_file.exists():
        data = torch.load(processed_file)
        return data.to(device)

    ratings = pd.read_csv(data_dir / "ratings.csv").drop(columns=["timestamp"])

    # Average tensors
    movie_avg = torch.tensor(
        ratings.groupby("movieId")["rating"].mean().values,
        dtype=torch.float32, device=device
    )
    user_avg = torch.tensor(
        ratings.groupby("userId")["rating"].mean().values,
        dtype=torch.float32, device=device
    )

    # Index mappings
    users      = sorted(ratings["userId"].unique())
    movies_ids = sorted(ratings["movieId"].unique())
    user2idx   = {u: i for i, u in enumerate(users)}
    movie2idx  = {m: i for i, m in enumerate(movies_ids)}
    num_movies = len(movies_ids)
    num_users  = len(users)

    # Movie genres
    movies_df       = pd.read_csv(data_dir / "movies.csv")
    movies_df["genres"] = movies_df["genres"].str.split("|")
    all_genres      = sorted({g for sub in movies_df["genres"] for g in sub})
    genre2idx       = {g: i for i, g in enumerate(all_genres, start=1)}
    num_genres      = len(genre2idx) + 1

    # Build genres_mat: (num_movies, max_g)
    genres_mat = torch.zeros((num_movies, max_g), dtype=torch.long)
    for _, row in movies_df.iterrows():
        mid = row["movieId"]
        if mid not in movie2idx:
            continue
        i = movie2idx[mid]
        g_idxs = [genre2idx[g] for g in row["genres"]][:max_g]
        genres_mat[i, :len(g_idxs)] = torch.tensor(g_idxs, dtype=torch.long)

    # User top‑k favorite genres: (num_users, user_fav_k)
    user_genre_counts = {u: [] for u in users}
    for _, row in ratings.iterrows():
        uid, mid = row["userId"], row["movieId"]
        if mid not in movie2idx: continue
        g_list = movies_df.loc[movies_df.movieId==mid, "genres"].iloc[0]
        user_genre_counts[uid].extend(g_list)

    user_genres_mat = torch.zeros((len(users), user_fav_k), dtype=torch.long)
    for u, lst in user_genre_counts.items():
        idx = user2idx[u]
        top_k = (
            pd.Series(lst)
            .map(genre2idx)
            .value_counts()
            .head(user_fav_k)
            .index
            .tolist()
    )
        user_genres_mat[idx, :len(top_k)] = torch.tensor(top_k, dtype=torch.long)

    # Split & Wrap
    train_df, val_df = train_test_split(ratings, test_size=test_size, random_state=seed)
    train_ds = ContentDataset(train_df, user2idx, movie2idx,
                genres_mat, user_genres_mat, movie_avg, user_avg)
    val_ds   = ContentDataset(val_df,   user2idx, movie2idx,
                genres_mat, user_genres_mat,movie_avg, user_avg)

    data = ContentData(
        train_ds=train_ds,
        val_ds=val_ds,
        user2idx=user2idx,
        movie2idx=movie2idx,
        genre2idx=genre2idx,
        genres_mat=genres_mat,
        user_genres_mat=user_genres_mat,
        movie_avg=movie_avg,
        user_avg=user_avg,
        num_users=num_users,
        num_movies=num_movies,
        num_genres=num_genres,
        movies_df=movies_df,
        ratings_df=ratings,
    )

    torch.save(data, processed_file)

    return data.to(device)
