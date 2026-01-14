import json
import torch
import random
import time
import numpy as np
from tqdm import tqdm
from model import LoveLiveTransformer
from game import LoveLiveGame

class Agent:
    def __init__(self, game):
        self.game = game

    def play(self, target_live_id):
        raise NotImplementedError

class PureAlgoAgent(Agent):
    def __init__(self, game):
        super().__init__(game)

    def play(self, target_live_id):
        self.game.start_game(target_live_id)
        turns = 0
        solved = False

        # Max turns safety
        for _ in range(20):
            turns += 1

            # Strategy:
            # 1. Check if candidates == 1
            if len(self.game.possible_live_ids) == 1:
                guess_live_id = list(self.game.possible_live_ids)[0]
                if self.game.guess_live(guess_live_id):
                    solved = True
                    break
                else:
                    pass

            # 2. Pick best song (Entropy)
            best_moves = self.game.get_best_moves(top_k=1)
            if best_moves:
                sid = best_moves[0][0]
            else:
                # Random fallback
                sid = random.choice(list(self.game.songs.keys()))

            # Pick artist (first available)
            aids = self.game.songs[sid]['artist_ids']
            aid = aids[0] if aids else list(self.game.artists.keys())[0]

            # Execute
            feedback = self.game.guess_song(sid, aid)
            self.game.prune_candidates(sid, aid, feedback)

        return solved, turns

class HybridAIAgent(Agent):
    def __init__(self, game, model_path='transformer_model.pth', mappings_path='mappings.json'):
        super().__init__(game)

        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
        self.song_to_idx = mappings['song_to_idx']
        self.artist_to_idx = mappings['artist_to_idx']
        self.live_to_idx = mappings['live_to_idx']
        self.idx_to_live = {v: k for k, v in self.live_to_idx.items()}

        num_songs = len(game.songs) + 1
        num_artists = len(game.artists) + 1
        num_feedback = 4
        num_lives = len(game.lives)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.model = LoveLiveTransformer(num_songs, num_artists, num_feedback, num_lives).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def play(self, target_live_id):
        self.game.start_game(target_live_id)
        turns = 0
        solved = False

        songs_seq = []
        artists_seq = []
        feedbacks_seq = []
        guessed_lives = set()

        for _ in range(20):
            turns += 1

            # 1. AI Live Prediction
            should_guess_live = False
            guess_live_id = None

            if len(songs_seq) > 0:
                s_in = torch.tensor([x + 1 for x in songs_seq], device=self.device).unsqueeze(1)
                a_in = torch.tensor([x + 1 for x in artists_seq], device=self.device).unsqueeze(1)
                f_in = torch.tensor([x + 1 for x in feedbacks_seq], device=self.device).unsqueeze(1)

                with torch.no_grad():
                    logits = self.model(s_in, a_in, f_in)
                    probs = torch.softmax(logits, dim=1).squeeze(0)

                # Pruning mask
                mask = torch.zeros_like(probs)
                possible_indices = [self.live_to_idx[lid] for lid in self.game.possible_live_ids]
                if possible_indices:
                    mask[possible_indices] = 1.0
                    probs = probs * mask
                    if probs.sum() > 0:
                        probs = probs / probs.sum()

                    top_idx = torch.argmax(probs).item()
                    top_prob = probs[top_idx].item()
                    top_lid = self.idx_to_live[top_idx]

                    # Threshold for risking a guess
                    if top_prob > 0.7 and top_lid not in guessed_lives:
                        should_guess_live = True
                        guess_live_id = top_lid

            # Also check absolute certainty
            if len(self.game.possible_live_ids) == 1:
                should_guess_live = True
                guess_live_id = list(self.game.possible_live_ids)[0]

            if should_guess_live:
                if self.game.guess_live(guess_live_id):
                    solved = True
                    break
                else:
                    guessed_lives.add(guess_live_id)
                    if guess_live_id in self.game.possible_live_ids:
                        self.game.possible_live_ids.remove(guess_live_id)

            # 2. Pick Song (Entropy)
            best_moves = self.game.get_best_moves(top_k=1)
            if best_moves:
                sid = best_moves[0][0]
            else:
                sid = random.choice(list(self.game.songs.keys()))

            aids = self.game.songs[sid]['artist_ids']
            aid = aids[0] if aids else list(self.game.artists.keys())[0]

            feedback = self.game.guess_song(sid, aid)
            self.game.prune_candidates(sid, aid, feedback)

            songs_seq.append(self.song_to_idx[sid])
            artists_seq.append(self.artist_to_idx[aid])
            feedbacks_seq.append(feedback)

        return solved, turns

def run_benchmark():
    game = LoveLiveGame()

    # Initialize agents
    algo_agent = PureAlgoAgent(game)
    try:
        ai_agent = HybridAIAgent(game)
        has_ai = True
    except Exception as e:
        print(f"Could not load AI Agent: {e}")
        has_ai = False

    # Test Setup
    num_games = 50
    test_live_ids = random.sample(game.live_ids, num_games)

    print(f"Starting Benchmark: {num_games} games")
    print("-" * 60)
    print(f"{'Agent':<15} | {'Avg Turns':<10} | {'Win Rate':<10} | {'Avg Time (s)':<12}")
    print("-" * 60)

    # Run Algo
    total_turns = 0
    total_time = 0
    wins = 0

    for lid in tqdm(test_live_ids, desc="Running Pure Algo"):
        start = time.time()
        solved, turns = algo_agent.play(lid)
        end = time.time()

        total_time += (end - start)
        total_turns += turns
        if solved: wins += 1

    print(f"{'Pure Algo':<15} | {total_turns/num_games:<10.2f} | {wins/num_games:<10.0%} | {total_time/num_games:<12.4f}")

    # Run AI
    if has_ai:
        total_turns = 0
        total_time = 0
        wins = 0

        for lid in tqdm(test_live_ids, desc="Running AI Hybrid"):
            start = time.time()
            solved, turns = ai_agent.play(lid)
            end = time.time()

            total_time += (end - start)
            total_turns += turns
            if solved: wins += 1

        print(f"{'AI Hybrid':<15} | {total_turns/num_games:<10.2f} | {wins/num_games:<10.0%} | {total_time/num_games:<12.4f}")

    print("-" * 60)

if __name__ == "__main__":
    run_benchmark()
