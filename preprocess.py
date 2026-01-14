import json
import os

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess():
    print("Loading data...")
    try:
        songs = load_json('data/song-info.json')
        artists = load_json('data/artists-info.json')
        lives_info = load_json('data/performance-info.json')
        setlists = load_json('data/performance-setlists.json')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print("Processing artists...")
    artist_map = {}
    for artist in artists:
        artist_map[artist['id']] = {
            'name': artist.get('name'),
            'englishName': artist.get('englishName'),
            # 'characters': artist.get('characters') # Not strictly needed for now
        }

    print("Processing songs...")
    song_map = {}
    for song in songs:
        # song['artists'] is a list of objects like {"id": "1", "variant": null}
        artist_ids = [a['id'] for a in song.get('artists', []) if a.get('id')]
        song_map[song['id']] = {
            'name': song.get('name'),
            'englishName': song.get('englishName'),
            'artist_ids': artist_ids
        }

    print("Processing lives...")
    processed_lives = {}

    # Filter lives that have setlist
    valid_lives_info = {l['id']: l for l in lives_info if l.get('hasSetlist')}

    for live_id, setlist_data in setlists.items():
        if live_id not in valid_lives_info:
            continue

        live_name = valid_lives_info[live_id].get('name')

        # Extract songs from setlist
        items = setlist_data.get('items', [])
        live_song_ids = []
        for item in items:
            if item.get('type') == 'song':
                sid = item.get('songId')
                if sid:
                    live_song_ids.append(sid)

        if not live_song_ids:
            continue

        # Determine artists in this live
        # We assume artists in the live are the union of artists of the songs in the setlist.
        # This is an approximation. A song might be covered by someone else.
        # But without explicit performer info per setlist item (which might exist but keeping it simple for now),
        # checking against song's original artist is the standard "Wordle" check requested.
        # "Is the artist right?" -> "Is the artist of the guessed song present in the target live?"
        # Ideally, we'd know who performed each song in the live.
        # Let's check setlist item structure again. It had "customSongName" but not explicit performer ID usually.
        # However, checking if the *Original Artist* of the guessed song is associated with any song in the live is a good proxy.
        # Or better: "Is the artist of the guessed song one of the performers in the live?"

        # Let's aggregate all artist IDs from all songs in the live.
        live_artist_ids = set()
        for sid in live_song_ids:
            if sid in song_map:
                for aid in song_map[sid]['artist_ids']:
                    live_artist_ids.add(aid)

        processed_lives[live_id] = {
            'name': live_name,
            'song_ids': list(set(live_song_ids)), # Unique songs
            'artist_ids': list(live_artist_ids)
        }

    print(f"Processed {len(processed_lives)} lives.")

    game_data = {
        'songs': song_map,
        'artists': artist_map,
        'lives': processed_lives
    }

    with open('game_data.json', 'w', encoding='utf-8') as f:
        json.dump(game_data, f, ensure_ascii=False, indent=2)
    print("Saved to game_data.json")

if __name__ == "__main__":
    preprocess()
