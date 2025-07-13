# 🎵 Spotify Hit Predictor

A comprehensive machine learning project that predicts song popularity using Spotify's audio features and metadata. This project implements both regression (exact popularity scores) and classification (popularity tiers) approaches using multiple algorithms.

## 🚀 Project Overview

This Spotify Hit Predictor uses machine learning to analyze what makes songs popular on Spotify. The system:

- **Collects data** from popular Spotify playlists using the Spotify Web API
- **Engineers features** from audio characteristics, artist data, and release timing
- **Trains multiple ML models** for both regression and classification tasks
- **Provides predictions** for new songs through an iOS mobile app

## 📁 Project Structure

```
SpotifyHitPredictor/
├── DataCollection/           # Data collection and feature engineering
│   ├── config.py            # Configuration and API settings
│   ├── spotify_client.py    # Spotify API wrapper with rate limiting
│   ├── data_collector.py    # Main data collection script
│   ├── feature_engineering.py # Feature engineering pipeline
│   ├── requirements.txt     # Python dependencies
│   └── data/               # Generated datasets (created after running)
├── ML/                     # Machine learning models and training
│   ├── models.py           # Model training and evaluation
│   ├── saved_models/       # Trained model files (created after training)
│   └── plots/             # Generated visualizations (created after training)
├── iOS/                   # iOS mobile app (future implementation)
└── README.md             # This file
```

## 🛠️ Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- Spotify Developer Account
- macOS (for iOS development later)

### 2. Spotify API Setup

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. Log in with your Spotify account
3. Click "Create App"
4. Fill in the app details:
   - **App Name**: Spotify Hit Predictor
   - **App Description**: ML system to predict song popularity
   - **Website**: http://localhost (or your website)
5. Copy your `Client ID` and `Client Secret`
6. Create your environment file:

```bash
# Copy the template and edit with your credentials
cp DataCollection/env_template.txt DataCollection/.env
# Then edit DataCollection/.env with your actual credentials
```

Your `.env` file should look like:
```bash
# DataCollection/.env
SPOTIFY_CLIENT_ID=your_actual_client_id_here
SPOTIFY_CLIENT_SECRET=your_actual_client_secret_here
```

**Important**: Never commit your `.env` file to version control - it's already in `.gitignore`.

#### Advanced Environment Configuration

The `env_template.txt` file includes optional advanced settings you can customize:

```bash
# Rate limiting (delay between API requests)
RATE_LIMIT_DELAY=0.1

# Maximum retries for failed requests  
MAX_RETRIES=3

# Market for search results (ISO country code)
MARKET=US

# Custom playlists (comma-separated playlist IDs)
CUSTOM_PLAYLISTS=37i9dQZEVXbMDoHDwVN2tF,37i9dQZEVXbLiRSasKsNU9

# Enable debug logging
DEBUG_LOGGING=false
```

### 3. Install Dependencies

```bash
cd DataCollection
pip install -r requirements.txt
```

## 🎯 Usage Guide

### Phase 1: Data Collection

```bash
cd DataCollection
python data_collector.py
```

This will:
- Collect tracks from popular Spotify playlists (Top 50, Viral 50, etc.)
- Gather audio features (tempo, energy, valence, etc.)
- Collect artist information and popularity metrics
- Calculate release recency and timing features
- Save raw dataset to `data/complete_dataset.csv`

### Phase 2: Feature Engineering

```bash
cd DataCollection
python feature_engineering.py
```

This creates advanced features:
- **Tempo categories** (slow/medium/fast)
- **Key-mode combinations** (C Major, G Minor, etc.)
- **Feature ratios** (energy/valence, dance/energy)
- **Artist popularity tiers** (indie/emerging/popular/superstar)
- **Release timing features** (season, holiday, popular months)
- **Interaction features** (party potential, uplifting score)

### Phase 3: Model Training

```bash
cd ML
python models.py
```

This trains multiple models:
- **Regression models**: Linear Regression, Random Forest, Gradient Boosting, SVR, Neural Network
- **Classification models**: Logistic Regression, Random Forest, Gradient Boosting, SVC, Neural Network
- Evaluates performance with cross-validation
- Generates feature importance analysis
- Saves trained models and visualizations

## 🔬 Technical Details

### Data Collection Strategy

- **Popular Playlists**: Global Top 50, Viral 50, Today's Top Hits, New Music Friday
- **Audio Features**: 13 audio characteristics from Spotify's API
- **Artist Data**: Popularity scores, follower counts, genre information
- **Time Analysis**: Release dates, recency, seasonal patterns
- **Rate Limiting**: Built-in handling of Spotify's API limits

### Feature Engineering

The system creates 50+ features from the raw data:

1. **Raw Audio Features** (13 features)
   - Acousticness, Danceability, Energy, Instrumentalness
   - Liveness, Loudness, Speechiness, Valence
   - Tempo, Key, Mode, Time Signature, Duration

2. **Derived Features** (35+ features)
   - Tempo categories and binary indicators
   - Key-mode combinations and popular key indicators
   - Feature ratios (energy/valence, dance/energy, etc.)
   - Duration categories (short/medium/long/radio-length)
   - Artist popularity and follower tiers
   - Release timing (season, holiday, popular months)
   - Loudness categories and normalization
   - Interaction features (party potential, chill score)

### Machine Learning Models

**Regression Task** (Predict 0-100 popularity score):
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression
- Multi-layer Perceptron

**Classification Task** (Predict popularity tier):
- Logistic Regression (baseline)
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Classifier
- Multi-layer Perceptron

### Model Evaluation

- **Cross-Validation**: 5-fold CV for robust performance estimates
- **Regression Metrics**: RMSE, R², Mean Absolute Error
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Feature Importance**: Analysis of which audio features matter most
- **Hyperparameter Tuning**: Grid search for optimal parameters

## 📊 Expected Results

Based on the comprehensive feature engineering and multiple algorithms, you can expect:

- **Regression R² scores**: 0.4-0.7 (depending on data quality and size)
- **Classification accuracy**: 60-80% for 4-tier popularity classification
- **Key insights**: Energy, danceability, and artist popularity typically show strong correlations with hit potential

## 🚀 Future Enhancements

### Planned Features
- **Genre-specific models** (pop vs. rock vs. hip-hop predictors)
- **Time-series analysis** (how popularity patterns change over months/years)
- **Advanced NLP** on track titles and artist names
- **Collaborative filtering** using user listening patterns
- **Real-time prediction API** for new releases

### iOS App Features
- Camera integration for playlist cover analysis
- Real-time audio feature extraction
- Social sharing of predictions
- Trending predictions dashboard
- Personalized recommendations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Data Schema

### Raw Track Data
```python
{
    'track_id': str,           # Spotify track ID
    'track_name': str,         # Song title
    'artist_names': str,       # Artist name(s)
    'popularity': int,         # Spotify popularity score (0-100)
    'release_date': str,       # Release date
    'duration_ms': int,        # Track duration in milliseconds
    # ... audio features
    'acousticness': float,     # 0.0 to 1.0
    'danceability': float,     # 0.0 to 1.0
    'energy': float,           # 0.0 to 1.0
    'valence': float,          # Musical positivity (0.0 to 1.0)
    'tempo': float,            # BPM
    # ... and more
}
```

### Engineered Features
```python
{
    # Original features plus:
    'tempo_category': str,     # 'Slow', 'Medium', 'Fast', etc.
    'key_mode': str,          # 'C_Major', 'G_Minor', etc.
    'feel_good_score': float, # Combined happiness metric
    'party_potential': float, # Danceability * Energy
    'artist_popularity_tier': str, # 'Indie', 'Popular', 'Superstar'
    'is_recent': bool,        # Released within 90 days
    # ... 40+ more engineered features
}
```

## 🎵 What Makes a Hit?

Based on typical analysis of Spotify data, successful songs often have:

- **High Energy** (0.6-0.8): Drives engagement and danceability
- **Moderate Tempo** (120-140 BPM): Sweet spot for most genres
- **High Valence** (0.5-0.8): Positive, uplifting feeling
- **Radio Length** (3-4 minutes): Optimal for streaming and radio play
- **Popular Artists** (>50 popularity): Existing fanbase helps
- **Strategic Timing**: Released during peak months (January, May, October)

## 🐛 Troubleshooting

### Common Issues

1. **Spotify API Rate Limits**
   - The system includes automatic retry logic
   - Data collection may take 30-60 minutes for full dataset

2. **Missing Audio Features**
   - Some tracks may not have audio features available
   - The system handles missing data with median imputation

3. **Memory Issues**
   - For large datasets (>10k tracks), consider processing in batches
   - Reduce the number of playlists in `config.py` if needed

4. **Model Training Time**
   - Neural networks and SVM models can be slow
   - Consider using fewer models for faster iteration

### Getting Help

- Check the logs for detailed error messages
- Ensure your Spotify API credentials are correct
- Verify you have enough API quota remaining
- Review the data quality in generated CSV files

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Spotify Web API](https://developer.spotify.com/documentation/web-api/) for providing rich music data
- [Spotipy](https://spotipy.readthedocs.io/) for the excellent Python Spotify client
- The music information retrieval community for research insights
- All the artists whose music makes this analysis possible

---

**Ready to predict the next hit? Let's make some music magic! 🎤✨**
