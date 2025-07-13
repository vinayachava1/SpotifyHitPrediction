import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import logging
from typing import List, Dict, Optional, Any
import requests
from config import (
    SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, 
    RATE_LIMIT_DELAY, MAX_RETRIES
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyClient:
    """
    Spotify API client with rate limiting and error handling
    """
    
    def __init__(self):
        """Initialize Spotify client with credentials"""
        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            raise ValueError("Spotify credentials not found. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your .env file")
        
        # Set up authentication
        auth_manager = SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        )
        
        self.sp = spotipy.Spotify(auth_manager=auth_manager)
        logger.info("Spotify client initialized successfully")
    
    def _retry_request(self, func, *args, **kwargs) -> Optional[Any]:
        """Retry a request with exponential backoff"""
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(RATE_LIMIT_DELAY)
                return func(*args, **kwargs)
            except spotipy.exceptions.SpotifyException as e:
                if e.http_status == 429:  # Rate limit exceeded
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                elif e.http_status == 404:
                    logger.warning(f"Resource not found: {e}")
                    return None
                else:
                    logger.error(f"Spotify API error: {e}")
                    if attempt == MAX_RETRIES - 1:
                        raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
        return None
    
    def get_playlist_tracks(self, playlist_id: str, limit: int = 100) -> List[Dict]:
        """
        Get all tracks from a playlist
        
        Args:
            playlist_id: Spotify playlist ID
            limit: Number of tracks to fetch per request (max 100)
            
        Returns:
            List of track dictionaries
        """
        tracks = []
        offset = 0
        
        while True:
            try:
                results = self._retry_request(
                    self.sp.playlist_tracks,
                    playlist_id=playlist_id,
                    offset=offset,
                    limit=limit,
                    fields='items(track(id,name,artists(name),album(name,release_date),popularity,duration_ms,external_urls)),next'
                )
                
                if not results or not results['items']:
                    break
                
                for item in results['items']:
                    if item['track'] and item['track']['id']:
                        tracks.append(item['track'])
                
                if not results['next']:
                    break
                    
                offset += limit
                
            except Exception as e:
                logger.error(f"Error fetching playlist tracks: {e}")
                break
        
        logger.info(f"Retrieved {len(tracks)} tracks from playlist {playlist_id}")
        return tracks
    
    def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        """
        Get audio features for multiple tracks
        
        Args:
            track_ids: List of track IDs
            
        Returns:
            List of audio feature dictionaries
        """
        features = []
        
        # Process in batches of 100 (Spotify API limit)
        for i in range(0, len(track_ids), 100):
            batch_ids = track_ids[i:i+100]
            
            try:
                batch_features = self._retry_request(
                    self.sp.audio_features,
                    tracks=batch_ids
                )
                
                if batch_features:
                    # Filter out None values
                    valid_features = [f for f in batch_features if f is not None]
                    features.extend(valid_features)
                    
            except Exception as e:
                logger.error(f"Error fetching audio features for batch: {e}")
                continue
        
        logger.info(f"Retrieved audio features for {len(features)} tracks")
        return features
    
    def get_track_details(self, track_ids: List[str]) -> List[Dict]:
        """
        Get detailed track information
        
        Args:
            track_ids: List of track IDs
            
        Returns:
            List of track detail dictionaries
        """
        tracks = []
        
        # Process in batches of 50 (Spotify API limit)
        for i in range(0, len(track_ids), 50):
            batch_ids = track_ids[i:i+50]
            
            try:
                batch_tracks = self._retry_request(
                    self.sp.tracks,
                    tracks=batch_ids
                )
                
                if batch_tracks and batch_tracks['tracks']:
                    tracks.extend(batch_tracks['tracks'])
                    
            except Exception as e:
                logger.error(f"Error fetching track details for batch: {e}")
                continue
        
        logger.info(f"Retrieved details for {len(tracks)} tracks")
        return tracks
    
    def search_tracks(self, query: str, limit: int = 50, market: str = 'US') -> List[Dict]:
        """
        Search for tracks using a query
        
        Args:
            query: Search query
            limit: Number of results to return
            market: Market for search results
            
        Returns:
            List of track dictionaries
        """
        try:
            results = self._retry_request(
                self.sp.search,
                q=query,
                type='track',
                limit=limit,
                market=market
            )
            
            if results and results['tracks']['items']:
                return results['tracks']['items']
                
        except Exception as e:
            logger.error(f"Error searching tracks: {e}")
            
        return []
    
    def get_artist_details(self, artist_ids: List[str]) -> List[Dict]:
        """
        Get artist details including popularity and genres
        
        Args:
            artist_ids: List of artist IDs
            
        Returns:
            List of artist dictionaries
        """
        artists = []
        
        # Process in batches of 50 (Spotify API limit)
        for i in range(0, len(artist_ids), 50):
            batch_ids = artist_ids[i:i+50]
            
            try:
                batch_artists = self._retry_request(
                    self.sp.artists,
                    artists=batch_ids
                )
                
                if batch_artists and batch_artists['artists']:
                    artists.extend(batch_artists['artists'])
                    
            except Exception as e:
                logger.error(f"Error fetching artist details for batch: {e}")
                continue
        
        logger.info(f"Retrieved details for {len(artists)} artists")
        return artists 