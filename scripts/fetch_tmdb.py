import os
import csv
import argparse
from time import sleep
from pathlib import Path

import requests
from tqdm import tqdm
from dotenv import load_dotenv

from ..config import ConfigLoader

RATE_LIMIT_DELAY = 0.02  # 50 req/sec
API_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"

cfg = ConfigLoader().load_config()

def parse_args() -> argparse.Namespace:
    data_dir = Path(cfg['dirs']['raw_data_dir'])
    parser = argparse.ArgumentParser(description="Fetch TMDB metadata for MovieLens movies")
    parser.add_argument("--links", type=Path, required=True, default=data_dir / 'links.csv', help="Path to MovieLens links.csv")
    parser.add_argument("--out", type=Path, required=True, default=data_dir / 'metadata.csv', help="Output path for metadata.csv")
    parser.add_argument("--rate", type=float, default=RATE_LIMIT_DELAY, help="Delay between requests in seconds, should not exceed 0.2")
    return parser.parse_args()


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise EnvironmentError("TMDB_API_KEY not found in .env file.")
    return api_key

def fetch_movie(tmdb_id: str, api_key: str) -> dict:
    url = API_URL.format(tmdb_id=tmdb_id)
    params = {"api_key": api_key, "language": "en-US"}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def main():
    args = parse_args()
    api_key = load_api_key()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Open input and output CSV files
    with open(args.links, newline='', encoding='utf-8') as links_file, \
        open(args.out, 'w', newline='', encoding='utf-8') as out_file:
        reader = csv.DictReader(links_file)
        fieldnames = [
            'movieId', 'title', 'year', 'genres',
            'overview', 'vote_average', 'vote_count', 'popularity'
        ]
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader, desc="Fetching TMDB metadata", unit=" movie "):
            movie_id = row['movieId']
            tmdb_id = row['tmdbId']

            try:
                data = fetch_movie(tmdb_id, api_key)
            except requests.HTTPError as e:
                print(f"Failed to fetch TMDB data for movieId {movie_id} (tmdbId={tmdb_id}): {e}")
                continue

            # Write to CSV
            writer.writerow({
                'movieId': movie_id,
                'title': data.get('title', ''),
                'year': data.get('release_date', '').split('-')[0] if data.get('release_date') else '',
                'genres': [g['name'] for g in data.get('genres', [])],
                'overview': data.get('overview', ''),
                'vote_average': data.get('vote_average', 0),
                'vote_count': data.get('vote_count', 0),
                'popularity': data.get('popularity', 0)
            })

            sleep(args.rate)

if __name__ == '__main__':
    main()
