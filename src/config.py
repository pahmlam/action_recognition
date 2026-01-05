from dataclasses import dataclass

@dataclass
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    drop_rate: float = 0.1
    attn_drop_rate: float = 0.0  # Thêm tham số này dựa trên logic code
    drop_path_rate: float = 0.1  # Thêm tham số này cho Stochastic Depth