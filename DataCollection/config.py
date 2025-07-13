import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Spotify API credentials
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

# Data collection settings (with environment variable overrides)
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 50))  # Number of tracks to process at once
RATE_LIMIT_DELAY = float(os.getenv('RATE_LIMIT_DELAY', 0.1))  # Delay between API calls in seconds
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))  # Maximum number of retries for failed requests
MARKET = os.getenv('MARKET', 'US')  # Market/region for search results
DEBUG_LOGGING = os.getenv('DEBUG_LOGGING', 'false').lower() == 'true'

# Popular playlist IDs for data collection
POPULAR_PLAYLISTS = {
    'global_top_50': '37i9dQZEVXbMDoHDwVN2tF',
    'global_viral_50': '37i9dQZEVXbLiRSasKsNU9',
    'us_top_50': '37i9dQZEVXbLRQDuF5jeBp',
    'us_viral_50': '37i9dQZEVXbKuaTI1Z1Afx',
    'today_top_hits': '37i9dQZF1DXcBWIGoYBM5M',
    'pop_rising': '37i9dQZF1DWUa8ZRTfalHk',
    'new_music_friday': '37i9dQZF1DX4JAvHpjipBk',
    'discover_weekly_example': '37i9dQZF1E35nOcOLnLJq3'  # This will vary per user
}

# Custom playlists from environment variable
CUSTOM_PLAYLISTS_ENV = os.getenv('CUSTOM_PLAYLISTS', '')
if CUSTOM_PLAYLISTS_ENV:
    custom_ids = [pid.strip() for pid in CUSTOM_PLAYLISTS_ENV.split(',') if pid.strip()]
    for i, playlist_id in enumerate(custom_ids):
        POPULAR_PLAYLISTS[f'custom_{i+1}'] = playlist_id

# Audio feature names from Spotify API
AUDIO_FEATURES = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'valence', 'tempo', 'key',
    'mode', 'time_signature', 'duration_ms'
]

# Output file paths
DATA_DIR = os.getenv('DATA_OUTPUT_DIR', 'data')
RAW_TRACKS_FILE = f'{DATA_DIR}/raw_tracks.csv'
PROCESSED_FEATURES_FILE = f'{DATA_DIR}/processed_features.csv'
PLAYLIST_TRACKS_FILE = f'{DATA_DIR}/playlist_tracks.csv' 