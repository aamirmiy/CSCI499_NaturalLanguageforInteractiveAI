import torch.nn as nn
import torch

EMBED_MAX_NORM = 1

class CBOW_Model(nn.Module):

    def __init__(self, vocab_size: int,embedding_dim):
        super(CBOW_Model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embed(inputs_)
        x = x.sum(axis=1)
        x = self.linear(x)
        return x