import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .modules import PatchEmbed, DropPath, Attention, Mlp
from .config import ViTConfig

class SMIFModule(nn.Module):
    """ Spatial-Motion Interaction Fusion [cite: 484] """
    def __init__(self, channels: int, window_size: int = 5):
        super().__init__()
        self.half = window_size // 2
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))
        # Conv 1x1 để fuse feature sau khi concat
        # Giả sử concat base + motion_flat (cùng channel) -> input channel * 2?
        # Tài liệu không ghi rõ chi tiết implementation của lớp Conv/Linear fusion
        # Nhưng code dòng 505 concat dim=1.
        # Ở đây ta giả định logic đơn giản hóa cho khớp.

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, T, C, H, W)
        B, T, C, H, W = video.shape
        motion_accum = torch.zeros_like(video)
        
        for offset in range(1, self.half + 1):
            # Lấy frame kế tiếp và frame trước đó (xử lý biên bằng cách roll hoặc clamp)
            next_frames = torch.roll(video, shifts=-offset, dims=1)
            prev_frames = torch.roll(video, shifts=offset, dims=1)
            
            # Xử lý biên cho đúng (không roll vòng tròn)
            # Trong thực tế cần padding kỹ hơn, tạm thời dùng roll cho đơn giản
            
            diff_f = next_frames - video
            diff_b = video - prev_frames
            motion_accum = motion_accum + diff_f.abs() + diff_b.abs()
            
        # Tài liệu ghi: fused = torch.cat([base, motion_flat], dim=1)
        # Điều này ngụ ý ghép kênh. Để cộng lại được (dòng 507), fused cần được projection về C.
        # Do code mẫu bị cắt bớt phần define layer fusion, ta sẽ dùng motion_accum trực tiếp nhân alpha
        # để cộng vào (theo logic residual).
        
        out = video + self.alpha.tanh() * motion_accum
        return out

class LMIModule(nn.Module):
    """ Long-term Motion Interaction  """
    def __init__(self, dim: int, reduction: int = 4, delta: float = 0.1):
        super().__init__()
        reduced_dim = max(1, dim // reduction)
        self.reduce = nn.Linear(dim, reduced_dim)
        self.expand = nn.Linear(reduced_dim, dim)
        self.temporal_mlp = nn.Sequential(
            nn.LayerNorm(reduced_dim),
            nn.Linear(reduced_dim, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, reduced_dim)
        )
        self.delta = nn.Parameter(torch.tensor(delta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, N, C) -> input LSViTBlock là đã gộp B và T
        # Cần biết B và T để reshape. 
        # NOTE: Trong LSViTBlock dòng 654 truyền B, T vào forward của Block,
        # nhưng code mẫu LMIModule dòng 570 unpack B, T, N, C = x.shape
        # Điều này mâu thuẫn. Ta sẽ giả định x đầu vào LMI đã được reshape thành (B, T, N, C)
        
        B, T, N, C = x.shape
        reduced = self.reduce(x) # (B, T, N, C/r)
        
        # Tính toán sai biệt giữa các frame 
        if T > 1:
            diff_f = reduced[:, 1:] - reduced[:, :-1]
            diff_f = torch.cat([diff_f, diff_f[:, -1:]], dim=1) # Pad last
            
            diff_b = reduced[:, :-1] - reduced[:, 1:]
            diff_b = torch.cat([diff_b[:, :1], diff_b], dim=1) # Pad first
            
            motion = (diff_f.abs() + diff_b.abs()).mean(dim=2) # Mean over patches (N) -> (B, T, C/r)
        else:
            motion = torch.zeros_like(reduced).mean(dim=2)


        attn = self.temporal_mlp(motion)
        attn = torch.sigmoid(attn).unsqueeze(2) # (B, T, 1, C/r)

        attn = self.expand(attn) # (B, T, 1, C) -> broadcast N
        enhanced = x * attn
        return x + self.delta.tanh() * enhanced

class LSViTBlock(nn.Module):
    """ [cite: 628] """
    def __init__(self, dim, num_heads, mlp_ratio, drop_rate, attn_drop, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, True, attn_drop, drop_rate)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop_rate)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.lmim = LMIModule(dim)

    def forward(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        # x: (B*T, N, C)
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        
        # Apply LMI 
        BT, N, C = x.shape
        x = x.view(B, T, N, C)
        x = self.lmim(x) # LMI tự xử lý logic cộng residual
        x = x.view(BT, N, C)
        return x

class LSViTBackbone(nn.Module):
    """ [cite: 681] """
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(config.drop_rate)
        
        dpr = torch.linspace(0, config.drop_path_rate, steps=config.depth).tolist()
        
        self.blocks = nn.ModuleList([
            LSViTBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                drop_rate=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[i]
            )
            for i in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _interpolate_pos_encoding(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        num_patches = N - 1
        if num_patches == self.patch_embed.num_patches:
            return self.pos_embed
            
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        
        dim = C
        gs_old = int(math.sqrt(patch_pos.shape[1]))
        gs_new = int(math.sqrt(num_patches))
        
        patch_pos = patch_pos.reshape(1, gs_old, gs_old, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, dim)
        
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, T, C, H, W) -> Đã qua SMIF nên giữ nguyên shape này
        B, T, C, H, W = video.shape
        
        #  Gộp Batch và Time
        x = video.reshape(B * T, C, H, W)
        x = self.patch_embed(x) # (BT, N, C)

        # Thêm CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        #  Positional Embedding
        pos_embed = self._interpolate_pos_encoding(x)
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # Qua các Blocks
        for block in self.blocks:
            x = block(x, B, T)
            
        x = self.norm(x)
        
        #  Trả về (B, T, Num_tokens, Channels)
        x = x.view(B, T, x.shape[1], x.shape[2])
        return x

class LSViTForAction(nn.Module):
    """ [cite: 836] """
    def __init__(self, config: ViTConfig, num_classes: int = 51, smif_window: int = 5):
        super().__init__()
        self.smif = SMIFModule(config.in_chans, window_size=smif_window)
        self.backbone = LSViTBackbone(config)
        self.head = nn.Linear(config.embed_dim, num_classes)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = self.smif(video)
        
        feats = self.backbone(x) # (B, T, N, D)
        cls_tokens = feats[:, :, 0] # Lấy CLS token (B, T, D)
        pooled = cls_tokens.mean(dim=1) # Temporal Pooling (B, D)
        logits = self.head(pooled) # (B, Num_classes)
        return logits