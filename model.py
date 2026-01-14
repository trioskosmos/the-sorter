import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class LoveLiveTransformer(nn.Module):
    def __init__(self, num_songs, num_artists, num_feedback_types, num_lives, d_model=64, nhead=4, num_layers=2):
        super(LoveLiveTransformer, self).__init__()

        self.song_embedding = nn.Embedding(num_songs, d_model)
        self.artist_embedding = nn.Embedding(num_artists, d_model)
        self.feedback_embedding = nn.Embedding(num_feedback_types, d_model)

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc_out = nn.Linear(d_model, num_lives)
        self.d_model = d_model

    def forward(self, song_seq, artist_seq, feedback_seq):
        # seq input: (seq_len, batch_size)

        # Embed inputs
        # Combine embeddings? Simply sum them.
        src = self.song_embedding(song_seq) + self.artist_embedding(artist_seq) + self.feedback_embedding(feedback_seq)

        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # Transformer Encoder
        # output: (seq_len, batch_size, d_model)
        output = self.transformer_encoder(src)

        # Pooling: Take the mean or the last token?
        # Typically use [CLS] token or mean. Let's use mean over sequence.
        # Transpose to (batch_size, seq_len, d_model)
        output = output.transpose(0, 1)
        # Mean pooling
        output = torch.mean(output, dim=1)

        # Classification
        logits = self.fc_out(output)
        return logits
