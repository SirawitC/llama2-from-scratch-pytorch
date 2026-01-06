import torch


def precompute_theta_pos_frequencies(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0
):
    # According to the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding" the embedding dimension must be even
    assert head_dim % 2 == 0, "head_dim must be even."
    # theta_i = 10000^(-2(i-1)/dim) for i in [1, 2, ..., dim/2]
    # Shape: (head_dim/2,)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (head_dim/2,)
    theta = 1 / (theta ** (theta_numerator / head_dim)).to(device)
    # Construct m which is a position index tensor
    # Shape: (seq_len,)
    m = torch.arange(seq_len, device=device)
    # Element-wise multiply each position with each theta value
    # Shape: (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).float()
    # Convert to complex numbers in polar form c = R * exp(i * m * theta) where R = 1
    # Shape: (seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex
