import json

def prune():
    print("Loading game_data.json...")
    with open('game_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    lives = data['lives']
    songs = data['songs']
    artists = data['artists']

    print(f"Original counts - Songs: {len(songs)}, Artists: {len(artists)}, Lives: {len(lives)}")

    used_song_ids = set()
    used_artist_ids = set()

    # Collect used IDs from lives
    for live in lives.values():
        for sid in live['song_ids']:
            used_song_ids.add(sid)
            # Also add artists of this song (from the song definition)
            # Note: live['artist_ids'] is the aggregated set of artists in the live,
            # but we should ensure we keep artists linked to the used songs too.
            if sid in songs:
                 for aid in songs[sid]['artist_ids']:
                     used_artist_ids.add(aid)

        for aid in live['artist_ids']:
            used_artist_ids.add(aid)

    # Filter songs
    pruned_songs = {sid: s for sid, s in songs.items() if sid in used_song_ids}

    # Filter artists
    pruned_artists = {aid: a for aid, a in artists.items() if aid in used_artist_ids}

    print(f"Pruned counts - Songs: {len(pruned_songs)}, Artists: {len(pruned_artists)}")

    data['songs'] = pruned_songs
    data['artists'] = pruned_artists

    with open('game_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Saved pruned game_data.json")

if __name__ == "__main__":
    prune()
