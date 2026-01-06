import torch
import torch.nn as nn
import torch.nn.functional as F
from setting.config import ModelArgs


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "vocab_size must be set."

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (batch_size, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one tokens at a time is supported."

        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        h = self.tok_embeddings(tokens)  

        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]


        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, vocab_size)
        output = self.output(h).float()  
        return output