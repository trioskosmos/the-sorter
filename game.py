import json
import random
import difflib

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

        # Check if song is in target live
        if song_id in self.target_live['song_ids']:
            # Song is correct. Check artist.
            if artist_id in self.target_live['artist_ids']:
                return 2 # Song & Artist Correct
            else:
                return 1 # Song Correct, Artist Incorrect
        else:
            return 0 # Song Incorrect

    def guess_live(self, live_id):
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

def play_cli():
    game = LoveLiveGame()
    game.start_game()
    print("Welcome to LoveLive Wordle!")
    print("Guess the Live Concert!")

    while True:
        mode = input("\n[S] Guess Song / [L] Guess Live / [Q] Quit: ").upper()
        if mode == 'Q':
            print(f"The answer was: {game.target_live['name']}")
            break

        elif mode == 'S':
            s_name = input("Song Name: ")
            a_name = input("Artist Name: ")

            sid = game.find_song_id(s_name)
            aid = game.find_artist_id(a_name)

            if not sid:
                print("Song not found.")
                continue
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

        elif mode == 'L':
            l_name = input("Live Name: ")
            lid = game.find_live_id(l_name)
            if not lid:
                print("Live not found.")
                continue

            print(f"Guessing Live: {game.lives[lid]['name']}")
            if game.guess_live(lid):
                print("Congratulations! You found the live!")
                break
            else:
                print("Incorrect Live.")

if __name__ == "__main__":
    play_cli()
