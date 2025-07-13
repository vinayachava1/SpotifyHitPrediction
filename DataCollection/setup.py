#!/usr/bin/env python3
"""
Setup script for Spotify Hit Predictor
This script helps users initialize the project and check their setup.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
        return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        '../ML/saved_models',
        '../ML/plots'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_env_file():
    """Create .env file from template"""
    env_file = Path('.env')
    template_file = Path('env_template.txt')
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if template_file.exists():
        print("📝 Creating .env file from template...")
        # Copy template to .env
        with open(template_file, 'r') as template:
            content = template.read()
        
        with open(env_file, 'w') as env:
            env.write(content)
        
        print("✅ Created .env file")
        print("⚠️  Please edit .env file with your Spotify API credentials")
        return False
    else:
        print("⚠️  Template file not found, creating basic .env file...")
        
        # Create basic template
        basic_template = """# Spotify API Credentials
# Get these from https://developer.spotify.com/dashboard/
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here
"""
        
        with open(env_file, 'w') as f:
            f.write(basic_template)
        
        print("✅ Created basic .env file")
        print("⚠️  Please edit .env file with your Spotify API credentials")
        return False

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("❌ .env file not found")
        return create_env_file()
    else:
        # Check if credentials are set
        with open('.env', 'r') as f:
            content = f.read()
            
        if 'your_spotify_client_id_here' in content or 'your_spotify_client_secret_here' in content:
            print("⚠️  .env file exists but credentials need to be updated")
            print("\n🔧 To get your Spotify API credentials:")
            print("1. Go to https://developer.spotify.com/dashboard/")
            print("2. Log in with your Spotify account")
            print("3. Click 'Create App'")
            print("4. Fill in app name: 'Spotify Hit Predictor'")
            print("5. Copy Client ID and Client Secret to your .env file")
            return False
        else:
            print("✅ .env file exists with credentials")
            return True

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        print("💡 Try running: pip install --upgrade pip")
        print("💡 Or try: pip install -r requirements.txt --user")
        return False

def test_spotify_connection():
    """Test Spotify API connection"""
    print("Testing Spotify API connection...")
    
    try:
        from spotify_client import SpotifyClient
        client = SpotifyClient()
        
        # Try a simple search to test the connection
        results = client.search_tracks("test", limit=1)
        
        print("✅ Spotify API connection successful")
        if results:
            print(f"✅ API test successful - found track: {results[0]['name']}")
        return True
    except Exception as e:
        print(f"❌ Spotify API connection failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your .env file has correct credentials")
        print("2. Verify credentials at https://developer.spotify.com/dashboard/")
        print("3. Make sure your app is not restricted or suspended")
        return False

def show_quick_tips():
    """Show helpful tips for getting started"""
    print("\n💡 QUICK TIPS:")
    print("=" * 50)
    print("🎵 Sample Spotify App Settings:")
    print("   App Name: Spotify Hit Predictor")
    print("   App Description: ML system to predict song popularity")
    print("   Website: http://localhost (or your website)")
    print("   Redirect URI: http://localhost:8080 (not needed for this project)")
    
    print("\n🔧 If you get rate limited:")
    print("   • The system has built-in retry logic")
    print("   • Data collection may take 30-60 minutes")
    print("   • You can reduce playlist count in config.py")
    
    print("\n📊 Expected data collection:")
    print("   • ~500-2000 tracks from popular playlists")
    print("   • 50+ engineered features per track")
    print("   • Artist data and audio features")
    
    print("\n🚀 After setup, try these commands:")
    print("   python data_collector.py      # Collect data")
    print("   python feature_engineering.py # Engineer features")
    print("   python example_usage.py       # See examples")

def main():
    """Main setup function"""
    print("🎵 Spotify Hit Predictor Setup")
    print("=" * 50)
    
    setup_steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", create_directories),
        ("Installing requirements", install_requirements),
        ("Setting up .env file", check_env_file),
    ]
    
    all_good = True
    
    for step_name, step_func in setup_steps:
        print(f"\n{step_name}...")
        if not step_func():
            all_good = False
    
    # Only test Spotify connection if env file is properly set up
    if all_good:
        print(f"\nTesting Spotify API connection...")
        if test_spotify_connection():
            print("\n🎉 Setup completed successfully!")
            print("\nNext steps:")
            print("1. Run: python data_collector.py")
            print("2. Run: python feature_engineering.py") 
            print("3. Run: cd ../ML && python models.py")
            print("4. Run: cd ../ML && python predict_hit.py")
        else:
            all_good = False
    
    if not all_good:
        print("\n⚠️  Setup incomplete. Please resolve the issues above.")
        show_quick_tips()
        print("\n🔍 Need help? Check README.md for detailed instructions")

if __name__ == "__main__":
    main() 