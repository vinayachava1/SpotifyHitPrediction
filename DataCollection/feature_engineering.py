import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging
from typing import List, Dict, Tuple, Optional
from config import AUDIO_FEATURES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering class for Spotify hit prediction
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def create_tempo_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tempo categories from continuous tempo values
        
        Args:
            df: DataFrame with tempo column
            
        Returns:
            DataFrame with tempo categories added
        """
        df = df.copy()
        
        # Define tempo ranges (BPM)
        df['tempo_category'] = pd.cut(
            df['tempo'], 
            bins=[0, 70, 100, 130, 160, float('inf')], 
            labels=['Very Slow', 'Slow', 'Medium', 'Fast', 'Very Fast']
        )
        
        # Create binary tempo features
        df['is_slow_tempo'] = (df['tempo'] < 100).astype(int)
        df['is_medium_tempo'] = ((df['tempo'] >= 100) & (df['tempo'] < 130)).astype(int)
        df['is_fast_tempo'] = (df['tempo'] >= 130).astype(int)
        
        return df
    
    def create_key_mode_combinations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create key-mode combination features
        
        Args:
            df: DataFrame with key and mode columns
            
        Returns:
            DataFrame with key-mode combinations added
        """
        df = df.copy()
        
        # Map musical keys to names
        key_names = {
            0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
            6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
        }
        
        # Map mode to names
        mode_names = {0: 'Minor', 1: 'Major'}
        
        # Create key and mode name columns
        df['key_name'] = df['key'].map(key_names)
        df['mode_name'] = df['mode'].map(mode_names)
        
        # Create key-mode combination
        df['key_mode'] = df['key_name'] + '_' + df['mode_name']
        
        # Create popular key combinations
        popular_keys = ['C_Major', 'G_Major', 'D_Major', 'A_Major', 'E_Major']
        df['is_popular_key'] = df['key_mode'].isin(popular_keys).astype(int)
        
        # Create major/minor binary features
        df['is_major'] = (df['mode'] == 1).astype(int)
        df['is_minor'] = (df['mode'] == 0).astype(int)
        
        return df
    
    def create_feature_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratios between audio features
        
        Args:
            df: DataFrame with audio features
            
        Returns:
            DataFrame with feature ratios added
        """
        df = df.copy()
        
        # Energy to valence ratio (how energetic vs happy)
        df['energy_valence_ratio'] = df['energy'] / (df['valence'] + 0.001)  # Add small value to avoid division by zero
        
        # Danceability to energy ratio
        df['dance_energy_ratio'] = df['danceability'] / (df['energy'] + 0.001)
        
        # Acousticness to energy ratio (acoustic vs electronic feel)
        df['acoustic_energy_ratio'] = df['acousticness'] / (df['energy'] + 0.001)
        
        # Speechiness to instrumentalness ratio
        df['speech_instrumental_ratio'] = df['speechiness'] / (df['instrumentalness'] + 0.001)
        
        # Liveness to energy ratio
        df['live_energy_ratio'] = df['liveness'] / (df['energy'] + 0.001)
        
        # Combined "feel good" score
        df['feel_good_score'] = (df['valence'] + df['danceability'] + df['energy']) / 3
        
        # Combined "chill" score
        df['chill_score'] = (df['acousticness'] + (1 - df['energy']) + (1 - df['tempo']/200)) / 3
        
        return df
    
    def create_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create duration-based features
        
        Args:
            df: DataFrame with duration_ms column
            
        Returns:
            DataFrame with duration features added
        """
        df = df.copy()
        
        # Convert to minutes if not already done
        if 'duration_minutes' not in df.columns:
            df['duration_minutes'] = df['duration_ms'] / 60000
        
        # Create duration categories
        df['duration_category'] = pd.cut(
            df['duration_minutes'], 
            bins=[0, 2, 3, 4, 5, float('inf')], 
            labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
        )
        
        # Binary duration features
        df['is_short_song'] = (df['duration_minutes'] < 3).astype(int)
        df['is_long_song'] = (df['duration_minutes'] > 4).astype(int)
        df['is_radio_length'] = ((df['duration_minutes'] >= 3) & (df['duration_minutes'] <= 4)).astype(int)
        
        return df
    
    def create_artist_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create artist-based features
        
        Args:
            df: DataFrame with artist information
            
        Returns:
            DataFrame with artist features added
        """
        df = df.copy()
        
        # Artist popularity tiers
        if 'artist_popularity' in df.columns:
            df['artist_popularity_tier'] = pd.cut(
                df['artist_popularity'], 
                bins=[0, 25, 50, 75, 100], 
                labels=['Unknown', 'Emerging', 'Popular', 'Superstar']
            )
            
            # Binary artist popularity features
            df['is_popular_artist'] = (df['artist_popularity'] > 50).astype(int)
            df['is_superstar_artist'] = (df['artist_popularity'] > 75).astype(int)
        
        # Artist followers tiers
        if 'artist_followers' in df.columns:
            df['artist_followers_tier'] = pd.cut(
                df['artist_followers'], 
                bins=[0, 10000, 100000, 1000000, float('inf')], 
                labels=['Indie', 'Emerging', 'Popular', 'Mainstream']
            )
        
        # Genre diversity
        if 'artist_genres_count' in df.columns:
            df['is_multi_genre'] = (df['artist_genres_count'] > 2).astype(int)
            df['is_niche_genre'] = (df['artist_genres_count'] <= 1).astype(int)
        
        return df
    
    def create_release_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create release timing features
        
        Args:
            df: DataFrame with release date information
            
        Returns:
            DataFrame with timing features added
        """
        df = df.copy()
        
        if 'release_month' in df.columns:
            # Season-based features
            df['release_season'] = df['release_month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
            
            # Popular release months (industry patterns)
            popular_months = [1, 5, 10, 11]  # January, May, October, November
            df['is_popular_release_month'] = df['release_month'].isin(popular_months).astype(int)
            
            # Holiday season releases
            df['is_holiday_release'] = df['release_month'].isin([11, 12]).astype(int)
            
            # Summer release
            df['is_summer_release'] = df['release_month'].isin([6, 7, 8]).astype(int)
        
        # Recency features
        if 'days_since_release' in df.columns:
            df['is_very_recent'] = (df['days_since_release'] <= 30).astype(int)
            df['is_recent'] = (df['days_since_release'] <= 90).astype(int)
            df['is_current_year'] = (df['days_since_release'] <= 365).astype(int)
        
        return df
    
    def create_loudness_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create loudness-based features
        
        Args:
            df: DataFrame with loudness column
            
        Returns:
            DataFrame with loudness features added
        """
        df = df.copy()
        
        if 'loudness' in df.columns:
            # Loudness categories (dB values are negative)
            df['loudness_category'] = pd.cut(
                df['loudness'], 
                bins=[-60, -15, -10, -5, 0], 
                labels=['Quiet', 'Moderate', 'Loud', 'Very Loud']
            )
            
            # Binary loudness features
            df['is_loud_track'] = (df['loudness'] > -10).astype(int)
            df['is_quiet_track'] = (df['loudness'] < -15).astype(int)
            
            # Normalized loudness (0-1 scale)
            df['loudness_normalized'] = (df['loudness'] + 60) / 60  # Assuming -60 to 0 range
            df['loudness_normalized'] = df['loudness_normalized'].clip(0, 1)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different audio characteristics
        
        Args:
            df: DataFrame with audio features
            
        Returns:
            DataFrame with interaction features added
        """
        df = df.copy()
        
        # Danceability * Energy (party potential)
        df['party_potential'] = df['danceability'] * df['energy']
        
        # Valence * Energy (uplifting potential)
        df['uplifting_potential'] = df['valence'] * df['energy']
        
        # Acousticness * Valence (feel-good acoustic)
        df['acoustic_happiness'] = df['acousticness'] * df['valence']
        
        # Speechiness * Energy (rap/spoken word energy)
        df['speech_energy'] = df['speechiness'] * df['energy']
        
        # Instrumentalness * Acousticness (acoustic instrumental)
        df['acoustic_instrumental'] = df['instrumentalness'] * df['acousticness']
        
        # Liveness * Energy (live performance energy)
        df['live_performance_energy'] = df['liveness'] * df['energy']
        
        return df
    
    def apply_all_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering techniques
        
        Args:
            df: Raw DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering...")
        
        df = df.copy()
        
        # Apply all feature engineering methods
        df = self.create_tempo_categories(df)
        df = self.create_key_mode_combinations(df)
        df = self.create_feature_ratios(df)
        df = self.create_duration_features(df)
        df = self.create_artist_features(df)
        df = self.create_release_timing_features(df)
        df = self.create_loudness_features(df)
        df = self.create_interaction_features(df)
        
        logger.info(f"Feature engineering complete. Dataset now has {len(df.columns)} features")
        return df
    
    def prepare_for_modeling(self, df: pd.DataFrame, target_column: str = 'popularity') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare the dataset for machine learning modeling
        
        Args:
            df: DataFrame with engineered features
            target_column: Name of the target variable
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = df.copy()
        
        # Separate features and target
        if target_column in df.columns:
            y = df[target_column]
            X = df.drop(columns=[target_column])
        else:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Remove non-feature columns
        columns_to_remove = [
            'track_id', 'track_name', 'artist_names', 'album_name', 
            'external_url', 'collection_date', 'artist_id', 'artist_name',
            'release_date', 'uri', 'track_href', 'analysis_url', 'type'
        ]
        
        X = X.drop(columns=[col for col in columns_to_remove if col in X.columns])
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        # One-hot encode categorical variables
        if len(categorical_columns) > 0:
            X = pd.get_dummies(X, columns=categorical_columns, prefix=categorical_columns)
        
        # Fill any remaining NaN values
        X = X.fillna(X.median())
        
        logger.info(f"Prepared dataset for modeling: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def get_feature_importance_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze feature correlations with popularity
        
        Args:
            df: DataFrame with features and popularity
            
        Returns:
            DataFrame with correlation analysis
        """
        # Calculate correlations with popularity
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_columns].corr()['popularity'].abs().sort_values(ascending=False)
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'feature': correlations.index,
            'correlation_with_popularity': correlations.values,
            'correlation_strength': pd.cut(
                correlations.values, 
                bins=[0, 0.1, 0.3, 0.5, 1.0], 
                labels=['Weak', 'Moderate', 'Strong', 'Very Strong']
            )
        })
        
        return analysis_df.head(20)  # Top 20 features

def main():
    """Main function for testing feature engineering"""
    import os
    from config import DATA_DIR
    
    # Load the dataset
    dataset_path = os.path.join(DATA_DIR, 'complete_dataset.csv')
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        
        # Initialize feature engineer
        engineer = FeatureEngineer()
        
        # Apply feature engineering
        engineered_df = engineer.apply_all_feature_engineering(df)
        
        # Prepare for modeling
        X, y = engineer.prepare_for_modeling(engineered_df)
        
        # Feature importance analysis
        importance_analysis = engineer.get_feature_importance_analysis(engineered_df)
        
        # Save results
        engineered_df.to_csv(os.path.join(DATA_DIR, 'engineered_features.csv'), index=False)
        importance_analysis.to_csv(os.path.join(DATA_DIR, 'feature_importance.csv'), index=False)
        
        print("Feature Engineering Results:")
        print(f"Original features: {len(df.columns)}")
        print(f"Engineered features: {len(engineered_df.columns)}")
        print(f"Modeling features: {len(X.columns)}")
        
        print("\nTop 10 Features by Correlation with Popularity:")
        print(importance_analysis.head(10))
        
    else:
        print(f"Dataset not found at {dataset_path}. Please run data collection first.")

if __name__ == "__main__":
    main() 