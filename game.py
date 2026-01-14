import json
import random
import difflib
import math
from collections import Counter

class LoveLiveGame:
    def __init__(self, data_path='game_data.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.lives = self.data['lives']
        self.songs = self.data['songs']
        self.artists = self.data['artists']

        self.live_ids = list(self.lives.keys())
        self.song_ids = list(self.songs.keys())
        self.artist_ids = list(self.artists.keys())

        self.target_live_id = None
        self.target_live = None

        # Candidates & History (Initialized here for safety, reset in start_game)
        self.possible_live_ids = set(self.live_ids)
        self.guessed_song_ids = set()
        self.guessed_live_ids = set()

        # Mappings for search
        self.song_name_map = {s['name']: sid for sid, s in self.songs.items()}
        self.artist_name_map = {a['name']: aid for aid, a in self.artists.items()}
        self.live_name_map = {l['name']: lid for lid, l in self.lives.items()}

    def start_game(self, target_id=None):
        if target_id and target_id in self.lives:
            self.target_live_id = target_id
        else:
            self.target_live_id = random.choice(self.live_ids)
        self.target_live = self.lives[self.target_live_id]

        # Reset state
        self.possible_live_ids = set(self.live_ids)
        self.guessed_song_ids = set()
        self.guessed_live_ids = set()

        return self.target_live_id

    def guess_song(self, song_id, artist_id):
        """
        Returns feedback code:
        0: Song NOT in live.
        1: Song in live, but Artist NOT in live.
        2: Song in live AND Artist in live.
        """
        if song_id not in self.songs:
            return -1 # Invalid song

        self.guessed_song_ids.add(song_id)

        # Check if song is in target live
        if song_id in self.target_live['song_ids']:
            # Song is correct. Check artist.
            if artist_id in self.target_live['artist_ids']:
                return 2 # Song & Artist Correct
            else:
                return 1 # Song Correct, Artist Incorrect
        else:
            return 0 # Song Incorrect

    def guess_song_only(self, song_id):
        """
        Used for Song-Only mode.
        Returns: (is_correct, matched_artist_ids)
        """
        if song_id not in self.songs:
            return False, []

        self.guessed_song_ids.add(song_id)

        if song_id in self.target_live['song_ids']:
            # Find artists in this live that are associated with this song
            live_artists = set(self.target_live['artist_ids'])
            song_artists = set(self.songs[song_id]['artist_ids'])

            # Intersection: Artists in the live who are known to perform this song
            matched = list(live_artists.intersection(song_artists))

            # Fallback: if intersection is empty (unexpected), return valid artists?
            # Or maybe just return all song_artists?
            # Let's return matched if any, else song_artists (maybe guest performer?)
            if not matched:
                 matched = list(song_artists)

            return True, matched
        else:
            return False, []

    def prune_candidates(self, song_id, artist_id, feedback):
        """
        Updates self.possible_live_ids based on feedback.
        Returns remaining count.
        """
        to_remove = set()
        for lid in self.possible_live_ids:
            live = self.lives[lid]
            has_song = song_id in live['song_ids']

            # For artist check, we only check if artist is in live's artist list
            has_artist = artist_id in live['artist_ids']

            if feedback == 0:
                # Song NOT in live
                if has_song:
                    to_remove.add(lid)
            elif feedback == 1:
                # Song IN live, Artist NOT in live
                if not has_song:
                    to_remove.add(lid)
                if has_artist:
                    to_remove.add(lid)
            elif feedback == 2:
                # Song IN live, Artist IN live
                if not has_song:
                    to_remove.add(lid)
                if not has_artist:
                    to_remove.add(lid)

        self.possible_live_ids -= to_remove
        return len(self.possible_live_ids)

    def guess_live(self, live_id):
        self.guessed_live_ids.add(live_id)
        return live_id == self.target_live_id

    def find_song_id(self, name):
        if name in self.song_name_map:
            return self.song_name_map[name]
        # Fuzzy search
        matches = difflib.get_close_matches(name, self.song_name_map.keys(), n=1, cutoff=0.6)
        if matches:
            return self.song_name_map[matches[0]]
        return None

    def find_artist_id(self, name):
        if name in self.artist_name_map:
            return self.artist_name_map[name]
        matches = difflib.get_close_matches(name, self.artist_name_map.keys(), n=1, cutoff=0.6)
        if matches:
            return self.artist_name_map[matches[0]]
        return None

    def find_live_id(self, name):
        if name in self.live_name_map:
            return self.live_name_map[name]
        matches = difflib.get_close_matches(name, self.live_name_map.keys(), n=1, cutoff=0.6)
        if matches:
            return self.live_name_map[matches[0]]
        return None

    def calculate_entropy(self, song_id):
        """
        Calculates the expected information gain (entropy) of guessing this song.
        Considers only 'Song Correct' vs 'Song Incorrect' to simplify,
        since Artist match depends on user choice (which we can't predict easily)
        or we assume 'Song Only' mode logic.

        Using binary outcome:
        - Song in live (Yes)
        - Song not in live (No)
        """
        candidate_count = len(self.possible_live_ids)
        if candidate_count <= 1:
            return 0.0

        # Count how many remaining lives have this song
        yes_count = 0
        for lid in self.possible_live_ids:
            if song_id in self.lives[lid]['song_ids']:
                yes_count += 1

        no_count = candidate_count - yes_count

        p_yes = yes_count / candidate_count
        p_no = no_count / candidate_count

        entropy = 0.0
        if p_yes > 0:
            entropy -= p_yes * math.log2(p_yes)
        if p_no > 0:
            entropy -= p_no * math.log2(p_no)

        return entropy

    def get_best_moves(self, top_k=5):
        """
        Returns list of (song_id, entropy_score)
        sorted by score descending.
        """
        scores = []

        # Optimization: Only consider songs that are present in at least one candidate live?
        # Or consider all songs?
        # Ideally, we want to split the candidate space.
        # A song not in ANY candidate live yields 0 entropy (p_no=1).
        # A song in ALL candidate lives yields 0 entropy (p_yes=1).
        # So we only need to check songs that appear in the candidate lives.

        relevant_songs = set()
        for lid in self.possible_live_ids:
            relevant_songs.update(self.lives[lid]['song_ids'])

        for sid in relevant_songs:
            if sid in self.guessed_song_ids:
                continue

            score = self.calculate_entropy(sid)
            scores.append((sid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

def play_cli():
    game = LoveLiveGame()
    game.start_game()
    print("Welcome to LoveLive Wordle!")

    # Select Mode
    print("\nSelect Game Mode:")
    print("1. Standard (Guess Song & Artist)")
    print("2. Song Only (Guess Song, Reveal Artist)")
    mode_choice = input("Choice [1/2]: ").strip()
    song_only_mode = (mode_choice == '2')

    # Pruning Toggle
    print("\nEnable Pruning Assistance? (Shows remaining candidates, warns on invalid guesses)")
    pruning_choice = input("Choice [y/n]: ").strip().lower()
    pruning_enabled = (pruning_choice == 'y')

    print("\nGuess the Live Concert!")

    while True:
        prompt = "\n[S] Guess Song / [L] Guess Live / [A] Analyze / [Q] Quit"
        if pruning_enabled:
            prompt += f" (Candidates: {len(game.possible_live_ids)})"
        prompt += ": "

        mode = input(prompt).upper()
        if mode == 'Q':
            print(f"The answer was: {game.target_live['name']}")
            break

        elif mode == 'A':
            print("Analyzing best moves (Entropy)...")
            best_moves = game.get_best_moves(top_k=5)
            print(f"{'Song Name':<40} | {'Score':<6}")
            print("-" * 50)
            for sid, score in best_moves:
                print(f"{game.songs[sid]['name']:<40} | {score:.4f}")
            print("-" * 50)

        elif mode == 'S':
            s_name = input("Song Name: ")
            sid = game.find_song_id(s_name)

            if not sid:
                print("Song not found.")
                continue

            if sid in game.guessed_song_ids:
                print("You already guessed this song!")
                continue

            if pruning_enabled:
                 # Check validity: Is this song in any remaining candidate live?
                 is_valid_guess = False
                 for lid in game.possible_live_ids:
                     if sid in game.lives[lid]['song_ids']:
                         is_valid_guess = True
                         break

                 if not is_valid_guess:
                     print("WARNING: This song is not in any of the remaining candidate lives!")
                     confirm = input("Guess anyway? [y/N]: ").strip().lower()
                     if confirm != 'y':
                         continue

            if song_only_mode:
                print(f"Guessing Song: {game.songs[sid]['name']}")
                is_correct, matched_artists = game.guess_song_only(sid)

                if is_correct:
                    artist_names = [game.artists[aid]['name'] for aid in matched_artists]
                    print(f">> CORRECT! Song is in the live.")
                    print(f"   Artists revealed: {', '.join(artist_names)}")

                    if pruning_enabled:
                         # Prune lives without this song AND without these artists
                         to_remove = set()
                         for lid in game.possible_live_ids:
                             l = game.lives[lid]
                             if sid not in l['song_ids']:
                                 to_remove.add(lid)
                             else:
                                 # Also check artists
                                 for aid in matched_artists:
                                     if aid not in l['artist_ids']:
                                         to_remove.add(lid)
                                         break

                         game.possible_live_ids -= to_remove
                         print(f"   (Candidates remaining: {len(game.possible_live_ids)})")

                else:
                    print(">> WRONG. (Song is not in the live)")
                    if pruning_enabled:
                        # Prune lives WITH this song
                        to_remove = set()
                        for lid in game.possible_live_ids:
                            if sid in game.lives[lid]['song_ids']:
                                to_remove.add(lid)
                        game.possible_live_ids -= to_remove
                        print(f"   (Candidates remaining: {len(game.possible_live_ids)})")

            else:
                a_name = input("Artist Name: ")
                aid = game.find_artist_id(a_name)

                if not aid:
                    print("Artist not found.")
                    continue

                print(f"Guessing: {game.songs[sid]['name']} / {game.artists[aid]['name']}")
                result = game.guess_song(sid, aid)

                if result == 2:
                    print(">> PERFECT MATCH! (Song & Artist are in the live)")
                elif result == 1:
                    print(">> SONG CORRECT! (But Artist is not in the live)")
                elif result == 0:
                    print(">> WRONG. (Song is not in the live)")

                if pruning_enabled:
                    rem = game.prune_candidates(sid, aid, result)
                    print(f"   (Candidates remaining: {rem})")

        elif mode == 'L':
            l_name = input("Live Name: ")
            lid = game.find_live_id(l_name)
            if not lid:
                print("Live not found.")
                continue

            if lid in game.guessed_live_ids:
                print("You already guessed this live!")
                continue

            print(f"Guessing Live: {game.lives[lid]['name']}")
            if game.guess_live(lid):
                print("Congratulations! You found the live!")
                break
            else:
                print("Incorrect Live.")
                if pruning_enabled:
                    if lid in game.possible_live_ids:
                        game.possible_live_ids.remove(lid)
                    print(f"   (Candidates remaining: {len(game.possible_live_ids)})")

if __name__ == "__main__":
    play_cli()
