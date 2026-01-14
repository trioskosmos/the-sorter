import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from tqdm import tqdm
from model import LoveLiveTransformer
from game import LoveLiveGame

class GameDataset(Dataset):
    def __init__(self, game, song_to_idx, artist_to_idx, live_to_idx, num_samples=10000, max_seq_len=20):
        self.game = game
        self.song_to_idx = song_to_idx
        self.artist_to_idx = artist_to_idx
        self.live_to_idx = live_to_idx
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.data = []

        self.all_song_ids = list(game.songs.keys())
        self.all_artist_ids = list(game.artists.keys())

        print(f"Pre-generating {num_samples} samples...")
        for _ in tqdm(range(num_samples)):
            self.data.append(self.generate_sample())

    def generate_sample(self):
        # 1. Pick a target live
        target_live_id = random.choice(self.game.live_ids)
        target_live = self.game.lives[target_live_id]

        # Set target live for feedback generation
        self.game.target_live_id = target_live_id
        self.game.target_live = target_live

        target_songs = target_live['song_ids']

        # 2. Generate guess sequence
        seq_len = random.randint(1, self.max_seq_len)

        songs_seq = []
        artists_seq = []
        feedbacks_seq = []

        for _ in range(seq_len):
            # Strategy: Mix of correct and incorrect guesses
            if target_songs and random.random() < 0.3:
                # Pick a song actually in the live
                guessed_song_id = random.choice(target_songs)
                # Pick correct artist or incorrect artist
                # Most of the time correct artist if song is correct (simulates knowing the song)
                song_artist_ids = self.game.songs[guessed_song_id]['artist_ids']
                if song_artist_ids and random.random() < 0.8:
                     guessed_artist_id = random.choice(song_artist_ids)
                else:
                     guessed_artist_id = random.choice(self.all_artist_ids)
            else:
                # Random guess
                guessed_song_id = random.choice(self.all_song_ids)
                guessed_artist_id = random.choice(self.all_artist_ids)

            feedback = self.game.guess_song(guessed_song_id, guessed_artist_id)

            # Map to indices
            songs_seq.append(self.song_to_idx[guessed_song_id])
            artists_seq.append(self.artist_to_idx[guessed_artist_id])
            feedbacks_seq.append(feedback) # 0, 1, 2

        # Pad to max_seq_len
        padded_songs = np.zeros(self.max_seq_len, dtype=int)
        padded_artists = np.zeros(self.max_seq_len, dtype=int)
        padded_feedbacks = np.zeros(self.max_seq_len, dtype=int)

        length = len(songs_seq)
        padded_songs[:length] = [s + 1 for s in songs_seq]
        padded_artists[:length] = [a + 1 for a in artists_seq]
        padded_feedbacks[:length] = [f + 1 for f in feedbacks_seq] # 0->1, 1->2, 2->3

        return (torch.tensor(padded_songs),
                torch.tensor(padded_artists),
                torch.tensor(padded_feedbacks),
                torch.tensor(self.live_to_idx[target_live_id]))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

def train():
    game = LoveLiveGame()

    # Create mappings
    song_to_idx = {sid: i for i, sid in enumerate(game.song_ids)}
    artist_to_idx = {aid: i for i, aid in enumerate(game.artist_ids)}
    live_to_idx = {lid: i for i, lid in enumerate(game.live_ids)}

    # Save mappings
    with open('mappings.json', 'w') as f:
        json.dump({
            'song_to_idx': song_to_idx,
            'artist_to_idx': artist_to_idx,
            'live_to_idx': live_to_idx
        }, f)

    dataset = GameDataset(game, song_to_idx, artist_to_idx, live_to_idx)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # +1 for padding
    num_songs = len(game.songs) + 1
    num_artists = len(game.artists) + 1
    num_feedback = 4 # 0(pad), 1(0), 2(1), 3(2)
    num_lives = len(game.lives)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = LoveLiveTransformer(num_songs, num_artists, num_feedback, num_lives).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for song_seq, artist_seq, fb_seq, target in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            song_seq, artist_seq, fb_seq, target = song_seq.to(device), artist_seq.to(device), fb_seq.to(device), target.to(device)

            # Input to model: (seq_len, batch_size)
            output = model(song_seq.T, artist_seq.T, fb_seq.T)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), 'transformer_model.pth')
    print("Model saved.")

if __name__ == "__main__":
    train()
