import json
import torch
import random
import numpy as np
from model import LoveLiveTransformer
from game import LoveLiveGame

def evaluate():
    print("Loading resources...")
    with open('mappings.json', 'r') as f:
        mappings = json.load(f)

    song_to_idx = mappings['song_to_idx']
    artist_to_idx = mappings['artist_to_idx']
    live_to_idx = mappings['live_to_idx']

    idx_to_live = {v: k for k, v in live_to_idx.items()}
    idx_to_song = {v: k for k, v in song_to_idx.items()}
    idx_to_artist = {v: k for k, v in artist_to_idx.items()}

    game = LoveLiveGame()

    # Model parameters must match train.py
    num_songs = len(game.songs) + 1
    num_artists = len(game.artists) + 1
    num_feedback = 4
    num_lives = len(game.lives)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LoveLiveTransformer(num_songs, num_artists, num_feedback, num_lives).to(device)
    model.load_state_dict(torch.load('transformer_model.pth', map_location=device))
    model.eval()

    # Start a simulation
    target_id = game.start_game()
    print(f"Target Live: {game.lives[target_id]['name']}")

    songs_seq = []
    artists_seq = []
    feedbacks_seq = []

    guessed_lives = set()

    max_turns = 20
    solved = False

    all_song_ids = list(game.songs.keys())

    for turn in range(max_turns):
        # Prepare input
        # Pad to max_len (20) used in training, or just use current seq?
        # Model expects (seq_len, batch_size)
        # We can pass current length seq.

        if len(songs_seq) == 0:
            # First turn: random guess or empty input?
            # Model trained on seq_len >= 1.
            # So first guess random.
            # Ideally "optimal" would mean picking a song that splits the space well initially.
            # Let's pick a very common song or just random.
            # Random for diversity.
            guess_song_id = random.choice(all_song_ids)
            # Pick an artist for this song
            artist_candidates = game.songs[guess_song_id]['artist_ids']
            guess_artist_id = random.choice(artist_candidates) if artist_candidates else random.choice(list(game.artists.keys()))

            print(f"Turn {turn+1}: First guess random -> {game.songs[guess_song_id]['name']}")
        else:
            # Use model to predict live
            # Pad inputs? Model uses positional encoding, so length matters.
            # Train used random seq len 1..20.
            # Just pass current seq.

            # Map indices + 1
            s_in = torch.tensor([x + 1 for x in songs_seq], device=device).unsqueeze(1) # (seq_len, 1)
            a_in = torch.tensor([x + 1 for x in artists_seq], device=device).unsqueeze(1)
            f_in = torch.tensor([x + 1 for x in feedbacks_seq], device=device).unsqueeze(1)

            with torch.no_grad():
                logits = model(s_in, a_in, f_in)
                probs = torch.softmax(logits, dim=1).squeeze(0) # (num_lives)

            # Sort predictions
            sorted_indices = torch.argsort(probs, descending=True)

            top_idx = sorted_indices[0]
            top_live_id = idx_to_live[top_idx.item()]
            top_prob = probs[top_idx]

            print(f"Turn {turn+1}: Top Prediction: {game.lives[top_live_id]['name']} ({top_prob.item():.4f})")

            if top_prob.item() > 0.7 and top_live_id not in guessed_lives:
                # Try guessing the live
                print(">> Guessing LIVE!")
                if game.guess_live(top_live_id):
                    print("CORRECT! Solved.")
                    solved = True
                    break
                else:
                    print("WRONG Live guess. Continuing...")
                    guessed_lives.add(top_live_id)

            # Choose next song
            # Pick the Top N lives
            top_k = 10
            top_indices = torch.topk(probs, k=top_k).indices.tolist()
            candidate_lives_ids = [idx_to_live[i] for i in top_indices]

            # Collect songs from these lives
            candidate_songs = []
            for lid in candidate_lives_ids:
                candidate_songs.extend(game.lives[lid]['song_ids'])

            # Count frequency
            from collections import Counter
            song_counts = Counter(candidate_songs)

            # Filter songs already guessed
            guessed_set = set(songs_seq) # indices
            # song_counts uses IDs

            best_song_id = None
            best_score = -1

            # Heuristic: Pick song closest to appearing in 50% of top candidates (Entropy maximization)
            target_count = top_k / 2

            for sid, count in song_counts.items():
                if song_to_idx[sid] in guessed_set:
                    continue

                score = -abs(count - target_count) # Maximize this (closest to 0 diff)
                if score > best_score:
                    best_score = score
                    best_song_id = sid

            if not best_song_id:
                best_song_id = random.choice(all_song_ids)

            guess_song_id = best_song_id
            # Pick likely artist for this song
            a_ids = game.songs[guess_song_id]['artist_ids']
            guess_artist_id = a_ids[0] if a_ids else list(game.artists.keys())[0]

            print(f"Guessing Song: {game.songs[guess_song_id]['name']}")

        # Execute guess
        feedback = game.guess_song(guess_song_id, guess_artist_id)
        print(f"Feedback: {feedback}")

        songs_seq.append(song_to_idx[guess_song_id])
        artists_seq.append(artist_to_idx[guess_artist_id])
        feedbacks_seq.append(feedback)

    if not solved:
        print("Failed to solve in max turns.")

if __name__ == "__main__":
    evaluate()
