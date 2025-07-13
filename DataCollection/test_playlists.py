#!/usr/bin/env python3
"""
Test Spotify Playlists Script
This script tests various playlist IDs to find which ones are accessible
and updates the configuration with working playlist IDs.
"""

import logging
from spotify_client import SpotifyClient
from config import POPULAR_PLAYLISTS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_playlist_access():
    """Test access to various Spotify playlists"""
    print("🎵 Testing Spotify Playlist Access")
    print("=" * 50)
    
    try:
        client = SpotifyClient()
        print("✅ Spotify client initialized successfully\n")
        
        # Test current playlists from config
        print("Testing playlists from current configuration:")
        print("-" * 45)
        
        working_playlists = {}
        failed_playlists = {}
        
        for name, playlist_id in POPULAR_PLAYLISTS.items():
            print(f"Testing {name}... ", end="")
            try:
                tracks = client.get_playlist_tracks(playlist_id)
                if tracks and len(tracks) > 0:
                    print(f"✅ Found {len(tracks)} tracks")
                    working_playlists[name] = playlist_id
                    # Show a sample track
                    if len(tracks) > 0:
                        sample_track = tracks[0]
                        print(f"   Sample: '{sample_track['name']}' by {sample_track['artists'][0]['name']}")
                else:
                    print("❌ No tracks found")
                    failed_playlists[name] = playlist_id
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                failed_playlists[name] = playlist_id
        
        print(f"\n📊 RESULTS:")
        print("-" * 20)
        print(f"✅ Working playlists: {len(working_playlists)}")
        print(f"❌ Failed playlists: {len(failed_playlists)}")
        
        if failed_playlists:
            print(f"\n❌ Failed Playlists:")
            for name, playlist_id in failed_playlists.items():
                print(f"   {name}: {playlist_id}")
        
        # Test some alternative known playlist IDs
        print(f"\n🔍 Testing alternative playlist IDs:")
        print("-" * 40)
        
        # These are more reliable playlist IDs that tend to work
        alternative_playlists = {
            'today_top_hits_alt': '37i9dQZF1DXcBWIGoYBM5M',
            'global_hits_alt': '37i9dQZF1DX0XUsuxWHRQd',
            'pop_rising_alt': '37i9dQZF1DWUa8ZRTfalHk',
            'viral_50_global_alt': '37i9dQZEVXbLiRSasKsNU9',
            'hot_100_alt': '37i9dQZF1DXcBWIGoYBM5M',
            'charts_alt': '37i9dQZF1DX0XUsuxWHRQd'
        }
        
        for name, playlist_id in alternative_playlists.items():
            if playlist_id not in [p for p in POPULAR_PLAYLISTS.values()]:
                print(f"Testing {name}... ", end="")
                try:
                    tracks = client.get_playlist_tracks(playlist_id)
                    if tracks and len(tracks) > 0:
                        print(f"✅ Found {len(tracks)} tracks")
                        working_playlists[name] = playlist_id
                        sample_track = tracks[0]
                        print(f"   Sample: '{sample_track['name']}' by {sample_track['artists'][0]['name']}")
                    else:
                        print("❌ No tracks found")
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
        
        # Generate updated configuration
        print(f"\n🔧 Generating updated configuration...")
        print("-" * 35)
        
        if working_playlists:
            print("Recommended playlist configuration:")
            print("```python")
            print("POPULAR_PLAYLISTS = {")
            for name, playlist_id in working_playlists.items():
                print(f"    '{name}': '{playlist_id}',")
            print("}")
            print("```")
            
            # Save to a file
            with open('working_playlists.py', 'w') as f:
                f.write("# Working Spotify Playlist IDs\n")
                f.write("# Generated on: " + str(__import__('datetime').datetime.now()) + "\n\n")
                f.write("WORKING_PLAYLISTS = {\n")
                for name, playlist_id in working_playlists.items():
                    f.write(f"    '{name}': '{playlist_id}',\n")
                f.write("}\n")
            
            print(f"\n💾 Saved working playlist IDs to 'working_playlists.py'")
            return working_playlists
        else:
            print("❌ No working playlists found. Please check your Spotify API credentials.")
            return {}
            
    except Exception as e:
        print(f"❌ Error initializing Spotify client: {e}")
        return {}

def test_individual_playlist(playlist_id):
    """Test a single playlist ID"""
    try:
        client = SpotifyClient()
        tracks = client.get_playlist_tracks(playlist_id)
        print(f"✅ Found {len(tracks)} tracks in playlist {playlist_id}")
        if tracks:
            print(f"First track: '{tracks[0]['name']}' by {tracks[0]['artists'][0]['name']}")
        return len(tracks) > 0
    except Exception as e:
        print(f"❌ Playlist {playlist_id} failed: {e}")
        return False

def find_working_playlists():
    """Find and return a set of working playlist IDs"""
    print("🔍 Searching for working Spotify playlists...")
    
    # Try various Spotify-curated playlist IDs
    test_playlists = {
        # Main charts and popular
        'today_top_hits': '37i9dQZF1DXcBWIGoYBM5M',
        'hot_country': '37i9dQZF1DX1lVhptIYRda',
        'rap_caviar': '37i9dQZF1DX0XUsuxWHRQd',
        'pop_rising': '37i9dQZF1DWUa8ZRTfalHk',
        'viral_50_global': '37i9dQZEVXbLiRSasKsNU9',
        
        # Discover and mood
        'discover_weekly_sample': '37i9dQZF1E35nOcOLnLJq3',
        'chill_hits': '37i9dQZF1DX4WYpdgoIcn6',
        'feel_good_pop': '37i9dQZF1DXdPec7aLTmlC',
        'acoustic_hits': '37i9dQZF1DX1gRalH1mWrP',
        
        # Alternative IDs for popular lists
        'global_top_alt1': '37i9dQZF1DXcBWIGoYBM5M',
        'global_top_alt2': '37i9dQZF1DX0XUsuxWHRQd'
    }
    
    working = {}
    
    try:
        client = SpotifyClient()
        
        for name, playlist_id in test_playlists.items():
            print(f"Testing {name}...", end=" ")
            try:
                tracks = client.get_playlist_tracks(playlist_id)
                if tracks and len(tracks) > 0:
                    print(f"✅ {len(tracks)} tracks")
                    working[name] = playlist_id
                else:
                    print("❌ Empty")
            except Exception as e:
                print(f"❌ Failed")
        
        return working
    
    except Exception as e:
        print(f"Error: {e}")
        return {}

if __name__ == "__main__":
    print("🎵 Spotify Playlist Tester")
    print("=" * 30)
    print("1. Test current configuration")
    print("2. Find working playlists")
    print("3. Test specific playlist ID")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        test_playlist_access()
    elif choice == "2":
        working = find_working_playlists()
        if working:
            print(f"\n✅ Found {len(working)} working playlists:")
            for name, pid in working.items():
                print(f"  {name}: {pid}")
        else:
            print("❌ No working playlists found")
    elif choice == "3":
        playlist_id = input("Enter playlist ID to test: ").strip()
        test_individual_playlist(playlist_id)
    else:
        print("Invalid choice, running default test...")
        test_playlist_access() 