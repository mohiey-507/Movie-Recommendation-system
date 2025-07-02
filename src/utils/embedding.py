from pathlib import Path

import faiss
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def embed_movie_overview(
        data_dir: str,
        sbert_model: str = 'all-MiniLM-L6-v2',
    ):
    data_dir = Path(data_dir)
    emb_dir = data_dir.parent / 'embeddings'
    emb_dir.mkdir(parents=True, exist_ok=True)

    emb_path = emb_dir / 'overview_emb.pt'

    # If embeddings already exist, load and return
    if emb_path.exists():
        overview_feats = torch.load(emb_path)
        embed_dim = overview_feats.shape[1]
        return overview_feats, embed_dim

    # Load metadata
    meta = pd.read_csv(data_dir / 'metadata.csv', usecols=['movieId', 'overview'])
    meta['overview'] = meta['overview']

    # Encode overviews
    sbert = SentenceTransformer(sbert_model)
    overview_emb = sbert.encode(
        meta['overview'].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Convert to tensor and save
    overview_feats = torch.tensor(overview_emb, dtype=torch.float32)
    torch.save(overview_feats, emb_path)
    embed_dim = overview_feats.shape[1]

    return overview_feats, embed_dim