from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for Q
    n_kv_heads: int | None = None # Number of heads for K and V
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_multiplier: float | None = None
    norm_eps: float = 1e-5

    max_seq_len: int = 2048
    max_batch_size: int = 32

    device: str = None