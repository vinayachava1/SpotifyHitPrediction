#!/usr/bin/env python3
"""
Example Usage Script for Spotify Hit Predictor
This script demonstrates how to use different components of the system.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Import our modules
from spotify_client import SpotifyClient
from data_collector import SpotifyDataCollector
from feature_engineering import FeatureEngineer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_spotify_client():
    """Example of using SpotifyClient directly"""
    print("\n" + "="*50)
    print("EXAMPLE 1: Using SpotifyClient")
    print("="*50)
    
    try:
        client = SpotifyClient()
        
        # Search for a popular song
        search_results = client.search_tracks("Shape of You Ed Sheeran", limit=1)
        if search_results:
            track = search_results[0]
            print(f"Found track: {track['name']} by {track['artists'][0]['name']}")
            print(f"Popularity: {track['popularity']}/100")
            
            # Get audio features
            features = client.get_audio_features([track['id']])
            if features:
                feature = features[0]
                print(f"Audio features:")
                print(f"  Danceability: {feature['danceability']:.2f}")
                print(f"  Energy: {feature['energy']:.2f}")
                print(f"  Valence: {feature['valence']:.2f}")
                print(f"  Tempo: {feature['tempo']:.0f} BPM")
        
    except Exception as e:
        print(f"Error: {e}")

def example_data_collection():
    """Example of collecting data from playlists"""
    print("\n" + "="*50)
    print("EXAMPLE 2: Data Collection from Playlists")
    print("="*50)
    
    try:
        collector = SpotifyDataCollector()
        
        # Collect from a smaller set of playlists for demo
        demo_playlists = {
            'global_top_50': '37i9dQZEVXbMDoHDwVN2tF',
            'today_top_hits': '37i9dQZF1DXcBWIGoYBM5M'
        }
        
        print("Collecting sample data (this may take a few minutes)...")
        dataset = collector.collect_full_dataset(demo_playlists)
        
        if not dataset.empty:
            print(f"Collected {len(dataset)} tracks")
            print(f"Features: {len(dataset.columns)}")
            
            print("\nSample tracks:")
            sample_tracks = dataset[['track_name', 'artist_names', 'popularity']].head()
            print(sample_tracks.to_string(index=False))
            
            print(f"\nPopularity distribution:")
            print(dataset['popularity'].describe())
            
            # Save sample data
            collector.save_data(dataset, 'sample_dataset.csv')
            print("Sample data saved to data/sample_dataset.csv")
        
    except Exception as e:
        print(f"Error: {e}")

def example_feature_engineering():
    """Example of feature engineering"""
    print("\n" + "="*50)
    print("EXAMPLE 3: Feature Engineering")
    print("="*50)
    
    try:
        # Load sample data if it exists
        import os
        from config import DATA_DIR
        
        sample_file = os.path.join(DATA_DIR, 'sample_dataset.csv')
        if os.path.exists(sample_file):
            df = pd.read_csv(sample_file)
            print(f"Loaded {len(df)} tracks from sample dataset")
        else:
            # Create synthetic data for demo
            print("Creating synthetic data for demo...")
            df = create_synthetic_data(100)
        
        # Initialize feature engineer
        engineer = FeatureEngineer()
        
        print(f"Original features: {len(df.columns)}")
        
        # Apply feature engineering
        engineered_df = engineer.apply_all_feature_engineering(df)
        print(f"Engineered features: {len(engineered_df.columns)}")
        
        # Show some new features
        new_features = [col for col in engineered_df.columns if col not in df.columns]
        print(f"\nSample of new features created: {new_features[:10]}")
        
        # Feature importance analysis
        importance = engineer.get_feature_importance_analysis(engineered_df)
        print(f"\nTop 5 features correlated with popularity:")
        print(importance.head()[['feature', 'correlation_with_popularity']].to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")

def create_synthetic_data(n_tracks: int) -> pd.DataFrame:
    """Create synthetic track data for demonstration"""
    np.random.seed(42)
    
    tracks = []
    for i in range(n_tracks):
        track = {
            'track_id': f'synthetic_{i}',
            'track_name': f'Song {i+1}',
            'artist_names': f'Artist {i%20 + 1}',
            'artist_ids': [f'artist_{i%20 + 1}'],
            'album_name': f'Album {i%30 + 1}',
            'release_date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
            'popularity': np.random.randint(0, 101),
            'duration_ms': np.random.randint(120000, 300000),  # 2-5 minutes
            'external_url': f'https://spotify.com/track/synthetic_{i}',
            'playlist_source': 'synthetic',
            'collection_date': datetime.now().isoformat(),
            
            # Audio features
            'acousticness': np.random.random(),
            'danceability': np.random.random(),
            'energy': np.random.random(),
            'instrumentalness': np.random.random(),
            'liveness': np.random.random(),
            'loudness': np.random.uniform(-60, 0),
            'speechiness': np.random.random(),
            'valence': np.random.random(),
            'tempo': np.random.uniform(60, 200),
            'key': np.random.randint(0, 12),
            'mode': np.random.randint(0, 2),
            'time_signature': np.random.choice([3, 4, 5]),
            
            # Artist data
            'artist_id': f'artist_{i%20 + 1}',
            'artist_name': f'Artist {i%20 + 1}',
            'artist_popularity': np.random.randint(0, 101),
            'artist_followers': np.random.randint(1000, 10000000),
            'artist_genres': 'pop, rock',
            'artist_genres_count': np.random.randint(1, 5)
        }
        tracks.append(track)
    
    return pd.DataFrame(tracks)

def example_track_analysis():
    """Example of analyzing specific tracks"""
    print("\n" + "="*50)
    print("EXAMPLE 4: Individual Track Analysis")
    print("="*50)
    
    try:
        client = SpotifyClient()
        
        # Analyze some popular songs
        songs_to_analyze = [
            "Shape of You Ed Sheeran",
            "Blinding Lights The Weeknd",
            "Watermelon Sugar Harry Styles"
        ]
        
        for song_query in songs_to_analyze:
            print(f"\nAnalyzing: {song_query}")
            print("-" * 30)
            
            # Search for the track
            results = client.search_tracks(song_query, limit=1)
            if results:
                track = results[0]
                track_id = track['id']
                
                # Get audio features
                features = client.get_audio_features([track_id])
                if features:
                    feature = features[0]
                    
                    print(f"Track: {track['name']}")
                    print(f"Artist: {', '.join([artist['name'] for artist in track['artists']])}")
                    print(f"Popularity: {track['popularity']}/100")
                    print(f"Release: {track['album']['release_date']}")
                    
                    # Analyze hit potential based on features
                    hit_score = calculate_hit_potential(feature, track['popularity'])
                    print(f"Hit Potential Score: {hit_score:.2f}")
                    
                    # Key audio features
                    print(f"Key Features:")
                    print(f"  Danceability: {feature['danceability']:.2f}")
                    print(f"  Energy: {feature['energy']:.2f}")
                    print(f"  Valence: {feature['valence']:.2f}")
                    print(f"  Tempo: {feature['tempo']:.0f} BPM")
        
    except Exception as e:
        print(f"Error: {e}")

def calculate_hit_potential(audio_features: dict, actual_popularity: int) -> float:
    """Simple hit potential calculation based on audio features"""
    # This is a simplified version - the real model is much more sophisticated
    weights = {
        'danceability': 0.25,
        'energy': 0.20,
        'valence': 0.15,
        'loudness': 0.10,  # Normalized
        'acousticness': -0.05,  # Too acoustic might be less popular
        'instrumentalness': -0.10,  # Vocals usually help
        'speechiness': -0.05,  # Too much speech might be less catchy
        'liveness': 0.05
    }
    
    score = 0
    for feature, weight in weights.items():
        if feature in audio_features:
            value = audio_features[feature]
            if feature == 'loudness':
                # Normalize loudness from [-60, 0] to [0, 1]
                value = (value + 60) / 60
                value = max(0, min(1, value))
            score += value * weight
    
    # Scale to 0-100
    return max(0, min(100, score * 100))

def example_genre_analysis():
    """Example of analyzing different genres"""
    print("\n" + "="*50)
    print("EXAMPLE 5: Genre-based Analysis")
    print("="*50)
    
    try:
        client = SpotifyClient()
        
        # Different genre examples
        genre_examples = {
            'Pop': ['Shape of You Ed Sheeran', 'Anti-Hero Taylor Swift'],
            'Hip-Hop': ['God\'s Plan Drake', 'HUMBLE. Kendrick Lamar'],
            'Rock': ['Thunderstruck AC/DC', 'Bohemian Rhapsody Queen'],
            'Electronic': ['Levels Avicii', 'Titanium David Guetta']
        }
        
        genre_analysis = {}
        
        for genre, songs in genre_examples.items():
            print(f"\nAnalyzing {genre} genre...")
            genre_features = []
            
            for song in songs:
                results = client.search_tracks(song, limit=1)
                if results:
                    track = results[0]
                    features = client.get_audio_features([track['id']])
                    if features:
                        feature = features[0]
                        feature['popularity'] = track['popularity']
                        feature['track_name'] = track['name']
                        genre_features.append(feature)
            
            if genre_features:
                # Calculate average features for this genre
                avg_features = {}
                for key in ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'popularity']:
                    avg_features[key] = np.mean([f[key] for f in genre_features])
                
                genre_analysis[genre] = avg_features
                
                print(f"Average {genre} characteristics:")
                print(f"  Danceability: {avg_features['danceability']:.2f}")
                print(f"  Energy: {avg_features['energy']:.2f}")
                print(f"  Valence: {avg_features['valence']:.2f}")
                print(f"  Tempo: {avg_features['tempo']:.0f} BPM")
                print(f"  Popularity: {avg_features['popularity']:.0f}/100")
        
        # Compare genres
        if len(genre_analysis) > 1:
            print(f"\nGenre Comparison:")
            print("-" * 30)
            for genre, features in genre_analysis.items():
                print(f"{genre:10} - Energy: {features['energy']:.2f}, Dance: {features['danceability']:.2f}, Pop: {features['popularity']:.0f}")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all examples"""
    print("🎵 Spotify Hit Predictor - Example Usage")
    print("This script demonstrates various features of the system")
    print("Note: Some examples require Spotify API credentials")
    
    examples = [
        ("Spotify Client Usage", example_spotify_client),
        ("Data Collection", example_data_collection),
        ("Feature Engineering", example_feature_engineering),
        ("Track Analysis", example_track_analysis),
        ("Genre Analysis", example_genre_analysis)
    ]
    
    print("\nWhich example would you like to run?")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    print("6. Run all examples")
    print("7. Exit")
    
    try:
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '7':
            print("Goodbye!")
            return
        elif choice == '6':
            for name, func in examples:
                print(f"\n{'='*60}")
                print(f"Running: {name}")
                print('='*60)
                try:
                    func()
                except Exception as e:
                    print(f"Error in {name}: {e}")
        elif choice.isdigit() and 1 <= int(choice) <= 5:
            name, func = examples[int(choice) - 1]
            print(f"\nRunning: {name}")
            func()
        else:
            print("Invalid choice!")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 