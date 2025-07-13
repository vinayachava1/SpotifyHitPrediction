#!/usr/bin/env python3
"""
Quick Start Script for Spotify Hit Predictor
This script guides you through the entire process step by step.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "🎵" + "="*58 + "🎵")
    print(f"  {title}")
    print("🎵" + "="*58 + "🎵")

def print_step(step_num, title):
    """Print a formatted step"""
    print(f"\n🎯 STEP {step_num}: {title}")
    print("-" * 50)

def wait_for_user():
    """Wait for user to press Enter"""
    input("\nPress Enter to continue...")

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n⚙️  {description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(f"Output: {result.stdout[:200]}...")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_requirements():
    """Check if all requirements are met"""
    print_step(1, "Checking Requirements")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    else:
        print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Check if .env file exists
    env_path = Path("DataCollection/.env")
    template_path = Path("DataCollection/env_template.txt")
    
    if not env_path.exists():
        print("❌ .env file not found in DataCollection/")
        
        # Try to create from template
        if template_path.exists():
            print("📝 Creating .env file from template...")
            with open(template_path, 'r') as template:
                content = template.read()
            with open(env_path, 'w') as env_file:
                env_file.write(content)
            print("✅ Created .env file from template")
        else:
            print("📝 Creating basic .env file...")
            basic_content = """# Spotify API Credentials
# Get these from https://developer.spotify.com/dashboard/
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here
"""
            with open(env_path, 'w') as env_file:
                env_file.write(basic_content)
            print("✅ Created basic .env file")
        
        print("\n🔧 To get your Spotify API credentials:")
        print("1. Go to https://developer.spotify.com/dashboard/")
        print("2. Log in with your Spotify account")
        print("3. Click 'Create App'")
        print("4. Fill in app name: 'Spotify Hit Predictor'")
        print("5. Edit DataCollection/.env with your Client ID and Client Secret")
        return False
    else:
        print("✅ .env file found")
    
    # Check credentials
    with open(env_path) as f:
        content = f.read()
        if "your_spotify_client_id_here" in content or "your_spotify_client_secret_here" in content:
            print("❌ Please update your .env file with real Spotify API credentials")
            print("💡 Edit DataCollection/.env and replace the placeholder values")
            return False
        else:
            print("✅ Spotify credentials configured")
    
    return True

def setup_environment():
    """Set up the Python environment"""
    print_step(2, "Setting Up Environment")
    
    # Create directories
    directories = [
        "DataCollection/data",
        "ML/saved_models", 
        "ML/plots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Install requirements
    if not run_command("cd DataCollection && pip install -r requirements.txt", 
                      "Installing Python packages"):
        return False
    
    return True

def collect_data():
    """Run data collection"""
    print_step(3, "Collecting Data from Spotify")
    print("This step will:")
    print("• Connect to Spotify API")
    print("• Collect tracks from popular playlists")
    print("• Gather audio features and artist data")
    print("• This may take 15-30 minutes depending on API limits")
    
    wait_for_user()
    
    success = run_command(
        "cd DataCollection && python data_collector.py",
        "Collecting Spotify data"
    )
    
    if success:
        # Check if data was created
        data_file = Path("DataCollection/data/complete_dataset.csv")
        if data_file.exists():
            print(f"✅ Data collection complete! Dataset saved to {data_file}")
            return True
        else:
            print("❌ Data collection completed but no dataset file found")
            return False
    
    return False

def engineer_features():
    """Run feature engineering"""
    print_step(4, "Engineering Features")
    print("This step will:")
    print("• Create tempo categories and key combinations")
    print("• Calculate feature ratios and interactions")
    print("• Add artist popularity and release timing features")
    print("• Generate 50+ features from raw data")
    
    wait_for_user()
    
    success = run_command(
        "cd DataCollection && python feature_engineering.py",
        "Engineering features"
    )
    
    if success:
        features_file = Path("DataCollection/data/engineered_features.csv")
        if features_file.exists():
            print(f"✅ Feature engineering complete! Features saved to {features_file}")
            return True
        else:
            print("❌ Feature engineering completed but no features file found")
            return False
    
    return False

def train_models():
    """Train machine learning models"""
    print_step(5, "Training Machine Learning Models")
    print("This step will:")
    print("• Train regression models (predict exact popularity)")
    print("• Train classification models (predict popularity tiers)")
    print("• Use Random Forest, Gradient Boosting, Neural Networks, etc.")
    print("• Evaluate models with cross-validation")
    print("• This may take 10-20 minutes")
    
    wait_for_user()
    
    success = run_command(
        "cd ML && python models.py",
        "Training machine learning models"
    )
    
    if success:
        models_dir = Path("ML/saved_models")
        if models_dir.exists() and any(models_dir.iterdir()):
            print(f"✅ Model training complete! Models saved to {models_dir}")
            return True
        else:
            print("❌ Model training completed but no model files found")
            return False
    
    return False

def test_prediction():
    """Test the prediction system"""
    print_step(6, "Testing Prediction System")
    print("Let's test the system with a popular song!")
    
    # Try to predict a well-known hit
    test_queries = [
        "Shape of You Ed Sheeran",
        "Blinding Lights The Weeknd", 
        "Watermelon Sugar Harry Styles"
    ]
    
    for query in test_queries:
        print(f"\n🎵 Testing with: {query}")
        
        # Create a simple test script
        test_script = f'''
import sys
sys.path.append('../DataCollection')
from ML.predict_hit import HitPredictor

try:
    predictor = HitPredictor()
    result = predictor.predict_track_by_search("{query}")
    
    print(f"Track: {{result['track_info']['name']}}")
    print(f"Artist: {{result['track_info']['artist']}}")
    print(f"Current Popularity: {{result['track_info']['current_popularity']}}/100")
    print(f"Predicted Popularity: {{result['predictions']['predicted_popularity_score']:.1f}}/100")
    print(f"Hit Potential: {{result['predictions']['hit_potential']}}")
    print("✅ Prediction successful!")
    
except Exception as e:
    print(f"❌ Prediction failed: {{e}}")
'''
        
        # Write and run test script
        with open("test_prediction.py", "w") as f:
            f.write(test_script)
        
        success = run_command("python test_prediction.py", f"Testing prediction for {query}")
        
        # Clean up
        if Path("test_prediction.py").exists():
            Path("test_prediction.py").unlink()
        
        if success:
            break
    
    return success

def show_next_steps():
    """Show what users can do next"""
    print_step(7, "What's Next?")
    
    print("🎉 Congratulations! Your Spotify Hit Predictor is ready!")
    print("\n🚀 Here's what you can do now:")
    
    print("\n1. 🔮 Make Predictions")
    print("   cd ML && python predict_hit.py")
    print("   • Predict hit potential for any song")
    print("   • Search by artist and song name")
    print("   • Get detailed audio feature analysis")
    
    print("\n2. 📊 Explore Examples")
    print("   cd DataCollection && python example_usage.py")
    print("   • See different ways to use the system")
    print("   • Analyze genres and track characteristics")
    print("   • Learn about audio features")
    
    print("\n3. 🔧 Customize Your Models")
    print("   • Edit DataCollection/config.py to add more playlists")
    print("   • Modify ML/models.py to try different algorithms")
    print("   • Create genre-specific models")
    
    print("\n4. 📱 Build the iOS App")
    print("   • The ML models are ready for integration")
    print("   • Use the saved models in your iOS app")
    print("   • Implement real-time prediction features")
    
    print("\n5. 📈 Advanced Analytics")
    print("   • Analyze trends over time")
    print("   • Compare different genres")
    print("   • Study what makes hits in specific markets")
    
    print("\n💡 Pro Tips:")
    print("• Run data collection regularly to get fresh data")
    print("• Experiment with different feature combinations")
    print("• Try collecting data from different regions/markets")
    print("• Use the batch prediction feature for analyzing playlists")

def main():
    """Main quick start process"""
    print_header("SPOTIFY HIT PREDICTOR - QUICK START")
    print("Welcome! This script will guide you through setting up")
    print("your complete Spotify hit prediction system.")
    print("\nThe process includes:")
    print("1. ✅ Check requirements")
    print("2. 🛠️  Set up environment") 
    print("3. 📥 Collect data from Spotify")
    print("4. ⚙️  Engineer features")
    print("5. 🤖 Train ML models")
    print("6. 🧪 Test predictions")
    print("7. 🚀 Show next steps")
    
    print(f"\nEstimated time: 45-90 minutes")
    print("(Most time is spent on data collection)")
    
    if input("\nReady to start? (y/n): ").lower() != 'y':
        print("Goodbye! Run this script again when you're ready.")
        return
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above and try again.")
        return
    
    # Step 2: Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed. Please check the errors above.")
        return
    
    # Step 3: Collect data
    if not collect_data():
        print("\n❌ Data collection failed. Please check your Spotify API credentials.")
        return
    
    # Step 4: Engineer features
    if not engineer_features():
        print("\n❌ Feature engineering failed. Please check the errors above.")
        return
    
    # Step 5: Train models
    if not train_models():
        print("\n❌ Model training failed. Please check the errors above.")
        return
    
    # Step 6: Test prediction
    print("\n🧪 Testing the prediction system...")
    if test_prediction():
        print("✅ Prediction system is working!")
    else:
        print("⚠️  Prediction test failed, but you can still use the system manually")
    
    # Step 7: Show next steps
    show_next_steps()
    
    print("\n" + "🎵" + "="*58 + "🎵")
    print("  🎉 SETUP COMPLETE! READY TO PREDICT HITS! 🎉")
    print("🎵" + "="*58 + "🎵")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user. You can resume anytime!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the error and try again.") 