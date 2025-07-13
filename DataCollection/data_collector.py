import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Set
import json
from spotify_client import SpotifyClient
from config import (
    POPULAR_PLAYLISTS, DATA_DIR, RAW_TRACKS_FILE, 
    PLAYLIST_TRACKS_FILE, AUDIO_FEATURES
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyDataCollector:
    """
    Main data collector for Spotify hit prediction dataset
    """
    
    def __init__(self):
        """Initialize the data collector"""
        self.client = SpotifyClient()
        self.ensure_data_directory()
        
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            logger.info(f"Created data directory: {DATA_DIR}")
    
    def collect_playlist_data(self, playlist_ids: Dict[str, str] = None) -> pd.DataFrame:
        """
        Collect tracks from multiple playlists
        
        Args:
            playlist_ids: Dictionary of playlist names and IDs
            
        Returns:
            DataFrame with track information
        """
        if playlist_ids is None:
            playlist_ids = POPULAR_PLAYLISTS
        
        all_tracks = []
        track_ids_seen = set()  # To avoid duplicates
        
        for playlist_name, playlist_id in playlist_ids.items():
            logger.info(f"Collecting tracks from {playlist_name}...")
            
            try:
                tracks = self.client.get_playlist_tracks(playlist_id)
                
                for track in tracks:
                    # Skip if we've already seen this track
                    if track['id'] in track_ids_seen:
                        continue
                    
                    track_ids_seen.add(track['id'])
                    
                    # Extract relevant information
                    track_info = {
                        'track_id': track['id'],
                        'track_name': track['name'],
                        'artist_names': ', '.join([artist['name'] for artist in track['artists']]),
                        'artist_ids': [artist['id'] for artist in track['artists']],
                        'album_name': track['album']['name'],
                        'release_date': track['album']['release_date'],
                        'popularity': track['popularity'],
                        'duration_ms': track['duration_ms'],
                        'external_url': track['external_urls']['spotify'],
                        'playlist_source': playlist_name,
                        'collection_date': datetime.now().isoformat()
                    }
                    
                    all_tracks.append(track_info)
                    
            except Exception as e:
                logger.error(f"Error collecting from playlist {playlist_name}: {e}")
                continue
        
        df = pd.DataFrame(all_tracks)
        logger.info(f"Collected {len(df)} unique tracks from {len(playlist_ids)} playlists")
        
        return df
    
    def collect_audio_features(self, track_ids: List[str]) -> pd.DataFrame:
        """
        Collect audio features for tracks
        
        Args:
            track_ids: List of Spotify track IDs
            
        Returns:
            DataFrame with audio features
        """
        logger.info(f"Collecting audio features for {len(track_ids)} tracks...")
        
        audio_features = self.client.get_audio_features(track_ids)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(audio_features)
        
        # Rename 'id' column to 'track_id' for consistency
        if 'id' in features_df.columns:
            features_df = features_df.rename(columns={'id': 'track_id'})
        
        logger.info(f"Retrieved audio features for {len(features_df)} tracks")
        return features_df
    
    def collect_artist_data(self, artist_ids: List[str]) -> pd.DataFrame:
        """
        Collect artist information including popularity and genres
        
        Args:
            artist_ids: List of Spotify artist IDs
            
        Returns:
            DataFrame with artist information
        """
        # Flatten and deduplicate artist IDs
        flat_artist_ids = []
        for artist_id_list in artist_ids:
            if isinstance(artist_id_list, list):
                flat_artist_ids.extend(artist_id_list)
            else:
                flat_artist_ids.append(artist_id_list)
        
        unique_artist_ids = list(set(flat_artist_ids))
        logger.info(f"Collecting artist data for {len(unique_artist_ids)} unique artists...")
        
        artists = self.client.get_artist_details(unique_artist_ids)
        
        artist_data = []
        for artist in artists:
            artist_info = {
                'artist_id': artist['id'],
                'artist_name': artist['name'],
                'artist_popularity': artist['popularity'],
                'artist_followers': artist['followers']['total'],
                'artist_genres': ', '.join(artist['genres']) if artist['genres'] else '',
                'artist_genres_count': len(artist['genres'])
            }
            artist_data.append(artist_info)
        
        df = pd.DataFrame(artist_data)
        logger.info(f"Retrieved artist data for {len(df)} artists")
        return df
    
    def calculate_release_recency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate how recent each track's release is
        
        Args:
            df: DataFrame with release_date column
            
        Returns:
            DataFrame with recency features added
        """
        df = df.copy()
        
        # Convert release_date to datetime
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        
        # Calculate days since release
        current_date = datetime.now()
        df['days_since_release'] = (current_date - df['release_date']).dt.days
        
        # Create recency categories
        df['recency_category'] = pd.cut(
            df['days_since_release'], 
            bins=[0, 30, 90, 365, 1825, float('inf')], 
            labels=['Very Recent', 'Recent', 'Somewhat Recent', 'Old', 'Very Old']
        )
        
        # Extract release year and month
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """
        Save DataFrame to CSV file
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} records to {filepath}")
    
    def load_existing_data(self, filename: str) -> pd.DataFrame:
        """
        Load existing data from CSV file
        
        Args:
            filename: Input filename
            
        Returns:
            DataFrame or empty DataFrame if file doesn't exist
        """
        filepath = os.path.join(DATA_DIR, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
        else:
            logger.info(f"File {filepath} doesn't exist, returning empty DataFrame")
            return pd.DataFrame()
    
    def collect_full_dataset(self, playlists: Dict[str, str] = None) -> pd.DataFrame:
        """
        Collect complete dataset with tracks, audio features, and artist data
        
        Args:
            playlists: Dictionary of playlist names and IDs
            
        Returns:
            Complete DataFrame with all features
        """
        logger.info("Starting full dataset collection...")
        
        # Step 1: Collect basic track information from playlists
        tracks_df = self.collect_playlist_data(playlists)
        
        if tracks_df.empty:
            logger.error("No tracks collected from playlists")
            return pd.DataFrame()
        
        # Step 2: Collect audio features
        track_ids = tracks_df['track_id'].tolist()
        audio_features_df = self.collect_audio_features(track_ids)
        
        # Step 3: Collect artist data
        artist_ids = tracks_df['artist_ids'].tolist()
        artists_df = self.collect_artist_data(artist_ids)
        
        # Step 4: Merge all data
        # Merge tracks with audio features
        complete_df = tracks_df.merge(
            audio_features_df, 
            on='track_id', 
            how='left'
        )
        
        # For artist data, we need to handle multiple artists per track
        # We'll use the first artist's data for now (can be improved later)
        tracks_df['primary_artist_id'] = tracks_df['artist_ids'].apply(
            lambda x: eval(x)[0] if isinstance(x, str) and x.startswith('[') else x[0] if isinstance(x, list) else x
        )
        
        complete_df = complete_df.merge(
            artists_df, 
            left_on='primary_artist_id', 
            right_on='artist_id', 
            how='left'
        )
        
        # Step 5: Calculate release recency
        complete_df = self.calculate_release_recency(complete_df)
        
        # Step 6: Clean and organize columns
        complete_df = self.clean_dataset(complete_df)
        
        logger.info(f"Complete dataset collected with {len(complete_df)} tracks and {len(complete_df.columns)} features")
        return complete_df
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and organize the dataset
        
        Args:
            df: Raw dataset
            
        Returns:
            Cleaned dataset
        """
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['track_id'])
        
        # Handle missing values
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in AUDIO_FEATURES:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with 'Unknown'
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna('Unknown')
        
        # Create popularity tiers for classification
        df['popularity_tier'] = pd.cut(
            df['popularity'], 
            bins=[0, 25, 50, 75, 100], 
            labels=['Low', 'Medium', 'High', 'Viral']
        )
        
        # Convert duration to minutes
        df['duration_minutes'] = df['duration_ms'] / 60000
        
        return df

def main():
    """Main function to run data collection"""
    collector = SpotifyDataCollector()
    
    try:
        # Collect full dataset
        dataset = collector.collect_full_dataset()
        
        if not dataset.empty:
            # Save the complete dataset
            collector.save_data(dataset, 'complete_dataset.csv')
            
            # Save individual components for later use
            collector.save_data(dataset[['track_id', 'track_name', 'artist_names', 'popularity', 'playlist_source']], 'tracks_summary.csv')
            
            # Print dataset summary
            print("\n" + "="*50)
            print("DATASET COLLECTION SUMMARY")
            print("="*50)
            print(f"Total tracks collected: {len(dataset)}")
            print(f"Total features: {len(dataset.columns)}")
            print(f"Date range: {dataset['release_date'].min()} to {dataset['release_date'].max()}")
            print(f"Popularity range: {dataset['popularity'].min()} to {dataset['popularity'].max()}")
            print(f"Average popularity: {dataset['popularity'].mean():.2f}")
            
            print("\nPopularity distribution:")
            print(dataset['popularity_tier'].value_counts())
            
            print("\nPlaylist sources:")
            print(dataset['playlist_source'].value_counts())
            
            print("\nTop audio features correlation with popularity:")
            audio_features_subset = [col for col in AUDIO_FEATURES if col in dataset.columns]
            correlations = dataset[audio_features_subset + ['popularity']].corr()['popularity'].sort_values(ascending=False)
            print(correlations.head(10))
            
        else:
            print("No data collected. Please check your Spotify API credentials and playlist IDs.")
            
    except Exception as e:
        logger.error(f"Error in main data collection: {e}")
        raise

if __name__ == "__main__":
    main() 