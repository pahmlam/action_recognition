import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """ Chuyển đổi ảnh thành các Patch Embeddings [cite: 384] """
    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        
        
        self.proj = nn.Conv2d(
            config.in_chans,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x) # (B, Embed_dim, H/P, W/P)
        # Flatten và transpose để có dạng (B, N, D)
        x = x.flatten(2).transpose(1, 2) 
        return x

class DropPath(nn.Module):
    """ Drop paths (Stochastic Depth) [cite: 439] """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        # Shape để broadcast: (batch, 1, 1) cho vector
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
        
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_() # Binarize
        
        output = x.div(keep_prob) * random_tensor
        return output

class Attention(nn.Module):
    """ Multi-Head Self-Attention [cite: 461] """
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, 
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        #
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B, Heads, N, Head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Tính toán Attention Score
        # (B, Heads, N, Head_dim) @ (B, Heads, Head_dim, N) -> (B, Heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Tổng hợp thông tin
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """ MLP Block cơ bản dùng trong Transformer """
    def __init__(self, in_features, mlp_ratio=4.0, drop=0.):
        super().__init__()
        hidden_features = int(in_features * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x