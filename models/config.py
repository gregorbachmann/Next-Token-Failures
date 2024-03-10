from dataclasses import dataclass

import torch


@dataclass
class PhiConfig:
    name: str = 'phi_2'
    block_size: int = 2048
    vocab_size: int = 51200  #
    n_layers: int = 32
    n_heads: int = 32
    n_embd: int = 2560
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_flash: bool = True
    cache: bool = True
    base: int = 10000
    rope_dim: int = int(0.4 * 2560 // n_heads)
    initializer_range: float = 0.02
    max_bsz: int = 16
    resid_drop: float = 0.1
    dtype = torch.bfloat16


Phi2Config = PhiConfig()
Phi1_5Config = PhiConfig(
    name='phi_1_5',
    n_embd=2048,
    n_layers=24,
    rope_dim=int(0.5 * 2048 // 32)
)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_flash: bool = True if torch.cuda.is_available() else False
    teacherless_token: int = None
    dtype = torch.bfloat16
    cache: bool = True
    max_bsz: int = 16


@dataclass
class PythiaConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_flash: bool = True if torch.cuda.is_available() else False
    teacherless_token: int = None
    dtype = torch.bfloat16
    cache: bool = True
    max_bsz: int = 16,
    base: int = 10000
    rope_dim: int = int(0.25 * n_embd // n_heads)



