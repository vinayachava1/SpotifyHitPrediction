#!/usr/bin/env python3
"""
Hit Prediction Script for Spotify Hit Predictor
This script allows you to predict the hit potential of individual tracks.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directories to path
sys.path.append('../DataCollection')

try:
    from spotify_client import SpotifyClient
    from feature_engineering import FeatureEngineer
    from models import SpotifyHitPredictor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running from the ML directory and all dependencies are installed.")
    sys.exit(1)

class HitPredictor:
    """
    Class to predict hit potential for individual tracks
    """
    
    def __init__(self):
        """Initialize the hit predictor"""
        self.spotify_client = SpotifyClient()
        self.feature_engineer = FeatureEngineer()
        self.regression_model = None
        self.classification_model = None
        
        # Load trained models if available
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        model_dir = Path('saved_models')
        
        if model_dir.exists():
            try:
                self.regression_model = SpotifyHitPredictor('regression')
                self.regression_model.load_models('saved_models')
                print("✅ Loaded regression models")
                
                self.classification_model = SpotifyHitPredictor('classification')
                self.classification_model.load_models('saved_models')
                print("✅ Loaded classification models")
                
            except Exception as e:
                print(f"⚠️  Could not load models: {e}")
                print("Please train models first by running: python models.py")
        else:
            print("⚠️  No trained models found. Please run: python models.py")
    
    def get_track_data(self, track_id: str) -> dict:
        """
        Get comprehensive track data from Spotify
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Dictionary with track data
        """
        try:
            # Get basic track info
            track_details = self.spotify_client.get_track_details([track_id])
            if not track_details:
                raise ValueError("Track not found")
            
            track = track_details[0]
            
            # Get audio features
            audio_features = self.spotify_client.get_audio_features([track_id])
            if not audio_features:
                raise ValueError("Audio features not available")
            
            features = audio_features[0]
            
            # Get artist data
            artist_ids = [artist['id'] for artist in track['artists']]
            artist_data = self.spotify_client.get_artist_details(artist_ids)
            
            # Combine all data
            track_data = {
                'track_id': track['id'],
                'track_name': track['name'],
                'artist_names': ', '.join([artist['name'] for artist in track['artists']]),
                'artist_ids': artist_ids,
                'album_name': track['album']['name'],
                'release_date': track['album']['release_date'],
                'popularity': track['popularity'],
                'duration_ms': track['duration_ms'],
                'external_url': track['external_urls']['spotify'],
                'playlist_source': 'user_input',
                'collection_date': pd.Timestamp.now().isoformat()
            }
            
            # Add audio features
            for key, value in features.items():
                if key != 'id':
                    track_data[key] = value
            
            # Add artist data (use first artist)
            if artist_data:
                artist = artist_data[0]
                track_data.update({
                    'artist_id': artist['id'],
                    'artist_name': artist['name'],
                    'artist_popularity': artist['popularity'],
                    'artist_followers': artist['followers']['total'],
                    'artist_genres': ', '.join(artist['genres']) if artist['genres'] else '',
                    'artist_genres_count': len(artist['genres'])
                })
            
            return track_data
            
        except Exception as e:
            raise ValueError(f"Error getting track data: {e}")
    
    def predict_track_by_id(self, track_id: str) -> dict:
        """
        Predict hit potential for a track by Spotify ID
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Dictionary with predictions
        """
        if not self.regression_model or not self.classification_model:
            raise ValueError("Models not loaded. Please train models first.")
        
        # Get track data
        track_data = self.get_track_data(track_id)
        
        # Convert to DataFrame
        df = pd.DataFrame([track_data])
        
        # Add release recency calculation
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        current_date = pd.Timestamp.now()
        df['days_since_release'] = (current_date - df['release_date']).dt.days
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
        
        # Apply feature engineering
        engineered_df = self.feature_engineer.apply_all_feature_engineering(df)
        
        # Prepare for prediction
        X_reg, _ = self.feature_engineer.prepare_for_modeling(engineered_df, 'popularity')
        
        # Create classification target
        engineered_df['popularity_tier'] = pd.cut(
            engineered_df['popularity'], 
            bins=[0, 25, 50, 75, 100], 
            labels=['Low', 'Medium', 'High', 'Viral']
        )
        X_clf, _ = self.feature_engineer.prepare_for_modeling(engineered_df, 'popularity_tier')
        
        # Make predictions
        regression_pred = self.regression_model.predict_single_track(X_reg.iloc[0].to_dict())
        classification_pred = self.classification_model.predict_single_track(X_clf.iloc[0].to_dict())
        
        # Compile results
        results = {
            'track_info': {
                'name': track_data['track_name'],
                'artist': track_data['artist_names'],
                'album': track_data['album_name'],
                'release_date': track_data['release_date'],
                'current_popularity': track_data['popularity'],
                'spotify_url': track_data['external_url']
            },
            'predictions': {
                'predicted_popularity_score': float(regression_pred),
                'predicted_popularity_tier': str(classification_pred),
                'hit_potential': self.calculate_hit_potential(regression_pred, track_data)
            },
            'audio_features': {
                'danceability': track_data.get('danceability', 0),
                'energy': track_data.get('energy', 0),
                'valence': track_data.get('valence', 0),
                'tempo': track_data.get('tempo', 0),
                'acousticness': track_data.get('acousticness', 0),
                'loudness': track_data.get('loudness', 0)
            },
            'insights': self.generate_insights(track_data, regression_pred)
        }
        
        return results
    
    def predict_track_by_search(self, query: str) -> dict:
        """
        Search for a track and predict its hit potential
        
        Args:
            query: Search query (e.g., "track:Shape of You artist:Ed Sheeran")
            
        Returns:
            Dictionary with predictions
        """
        # Search for tracks
        search_results = self.spotify_client.search_tracks(query, limit=1)
        
        if not search_results:
            raise ValueError(f"No tracks found for query: {query}")
        
        track = search_results[0]
        track_id = track['id']
        
        return self.predict_track_by_id(track_id)
    
    def calculate_hit_potential(self, predicted_score: float, track_data: dict) -> str:
        """
        Calculate hit potential category
        
        Args:
            predicted_score: Predicted popularity score
            track_data: Track information
            
        Returns:
            Hit potential category
        """
        artist_boost = 1.0
        if track_data.get('artist_popularity', 0) > 75:
            artist_boost = 1.2
        elif track_data.get('artist_popularity', 0) > 50:
            artist_boost = 1.1
        
        adjusted_score = predicted_score * artist_boost
        
        if adjusted_score >= 80:
            return "🔥 VIRAL POTENTIAL"
        elif adjusted_score >= 65:
            return "🚀 HIGH HIT POTENTIAL"
        elif adjusted_score >= 45:
            return "📈 MODERATE HIT POTENTIAL"
        elif adjusted_score >= 25:
            return "💭 LOW HIT POTENTIAL"
        else:
            return "😴 NICHE APPEAL"
    
    def generate_insights(self, track_data: dict, predicted_score: float) -> list:
        """
        Generate insights about the track's hit potential
        
        Args:
            track_data: Track information
            predicted_score: Predicted popularity score
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Energy insights
        energy = track_data.get('energy', 0)
        if energy > 0.7:
            insights.append("⚡ High energy - great for workouts and parties")
        elif energy < 0.3:
            insights.append("🌙 Low energy - perfect for chill/relaxing playlists")
        
        # Danceability insights
        danceability = track_data.get('danceability', 0)
        if danceability > 0.7:
            insights.append("💃 Highly danceable - club and party potential")
        
        # Valence insights
        valence = track_data.get('valence', 0)
        if valence > 0.7:
            insights.append("😊 Very positive/happy vibe - feel-good factor")
        elif valence < 0.3:
            insights.append("😢 Melancholic feel - good for emotional playlists")
        
        # Tempo insights
        tempo = track_data.get('tempo', 0)
        if 120 <= tempo <= 140:
            insights.append("🎵 Perfect tempo for mainstream appeal")
        elif tempo > 160:
            insights.append("🏃‍♂️ Fast tempo - great for high-energy activities")
        
        # Artist popularity
        artist_pop = track_data.get('artist_popularity', 0)
        if artist_pop > 75:
            insights.append("🌟 Superstar artist - built-in fanbase advantage")
        elif artist_pop < 25:
            insights.append("🆕 Emerging artist - viral potential if track catches on")
        
        # Release timing
        release_month = pd.to_datetime(track_data.get('release_date', '')).month
        if release_month in [1, 5, 10, 11]:
            insights.append("📅 Released in peak month - timing advantage")
        
        return insights
    
    def batch_predict(self, track_ids: list) -> pd.DataFrame:
        """
        Predict hit potential for multiple tracks
        
        Args:
            track_ids: List of Spotify track IDs
            
        Returns:
            DataFrame with predictions
        """
        results = []
        
        for track_id in track_ids:
            try:
                prediction = self.predict_track_by_id(track_id)
                results.append({
                    'track_id': track_id,
                    'track_name': prediction['track_info']['name'],
                    'artist': prediction['track_info']['artist'],
                    'current_popularity': prediction['track_info']['current_popularity'],
                    'predicted_popularity': prediction['predictions']['predicted_popularity_score'],
                    'predicted_tier': prediction['predictions']['predicted_popularity_tier'],
                    'hit_potential': prediction['predictions']['hit_potential']
                })
            except Exception as e:
                print(f"Error predicting {track_id}: {e}")
                continue
        
        return pd.DataFrame(results)

def main():
    """Main function for interactive prediction"""
    print("🎵 Spotify Hit Predictor")
    print("=" * 50)
    
    try:
        predictor = HitPredictor()
        
        while True:
            print("\nOptions:")
            print("1. Predict by Spotify track ID")
            print("2. Search and predict")
            print("3. Batch predict from file")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                track_id = input("Enter Spotify track ID: ").strip()
                try:
                    result = predictor.predict_track_by_id(track_id)
                    print_prediction_result(result)
                except Exception as e:
                    print(f"Error: {e}")
            
            elif choice == '2':
                query = input("Enter search query (e.g., 'Shape of You Ed Sheeran'): ").strip()
                try:
                    result = predictor.predict_track_by_search(query)
                    print_prediction_result(result)
                except Exception as e:
                    print(f"Error: {e}")
            
            elif choice == '3':
                filename = input("Enter filename with track IDs (one per line): ").strip()
                try:
                    with open(filename, 'r') as f:
                        track_ids = [line.strip() for line in f if line.strip()]
                    
                    results_df = predictor.batch_predict(track_ids)
                    output_file = 'batch_predictions.csv'
                    results_df.to_csv(output_file, index=False)
                    print(f"Batch predictions saved to {output_file}")
                    print(results_df.to_string(index=False))
                    
                except Exception as e:
                    print(f"Error: {e}")
            
            elif choice == '4':
                print("Thanks for using Spotify Hit Predictor! 🎤")
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        print("Please ensure models are trained and Spotify API is configured.")

def print_prediction_result(result: dict):
    """Print formatted prediction result"""
    info = result['track_info']
    pred = result['predictions']
    audio = result['audio_features']
    insights = result['insights']
    
    print("\n" + "="*60)
    print("🎵 TRACK INFORMATION")
    print("="*60)
    print(f"Track: {info['name']}")
    print(f"Artist: {info['artist']}")
    print(f"Album: {info['album']}")
    print(f"Release Date: {info['release_date']}")
    print(f"Current Popularity: {info['current_popularity']}/100")
    print(f"Spotify URL: {info['spotify_url']}")
    
    print("\n" + "="*60)
    print("🔮 PREDICTIONS")
    print("="*60)
    print(f"Predicted Popularity Score: {pred['predicted_popularity_score']:.1f}/100")
    print(f"Predicted Tier: {pred['predicted_popularity_tier']}")
    print(f"Hit Potential: {pred['hit_potential']}")
    
    print("\n" + "="*60)
    print("🎶 AUDIO FEATURES")
    print("="*60)
    print(f"Danceability: {audio['danceability']:.2f}")
    print(f"Energy: {audio['energy']:.2f}")
    print(f"Valence (Positivity): {audio['valence']:.2f}")
    print(f"Tempo: {audio['tempo']:.0f} BPM")
    print(f"Acousticness: {audio['acousticness']:.2f}")
    print(f"Loudness: {audio['loudness']:.1f} dB")
    
    if insights:
        print("\n" + "="*60)
        print("💡 INSIGHTS")
        print("="*60)
        for insight in insights:
            print(f"• {insight}")
    
    print("="*60)

if __name__ == "__main__":
    main() 