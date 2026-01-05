import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import random
import re
from torchvision import transforms
import torchvision.transforms.functional as TF

class VideoTransform:
    """ [cite: 1083] """
    def __init__(self, image_size: int, is_train: bool = True):
        self.image_size = image_size
        self.is_train = is_train
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (T, C, H, W)
        if self.is_train:
            # Code augmentation 
            h, w = frames.shape[-2:]
            scale = random.uniform(0.8, 1.0)
            new_h, new_w = int(h * scale), int(w * scale)
            frames = TF.resize(frames, [new_h, new_w], interpolation=transforms.InterpolationMode.BILINEAR)
            
            i = random.randint(0, max(0, new_h - self.image_size))
            j = random.randint(0, max(0, new_w - self.image_size))
            frames = TF.crop(frames, i, j, min(self.image_size, new_h), min(self.image_size, new_w))
            frames = TF.resize(frames, [self.image_size, self.image_size]) # Ensure size
            
            if random.random() < 0.5:
                frames = TF.hflip(frames)
                
            if random.random() < 0.3:
                # Apply jitter per frame or batch? Usually per video batch consistent
                # Simplified implementation logic
                frames = TF.adjust_brightness(frames, random.uniform(0.9, 1.1))
                
        else:
             frames = TF.resize(frames, [self.image_size, self.image_size], interpolation=transforms.InterpolationMode.BILINEAR)
        
        # Normalize
        # frames shape: T, C, H, W
        normalized = torch.stack([TF.normalize(frame, self.mean, self.std) for frame in frames])
        return normalized

class HMDB51Dataset(Dataset):
    """ [cite: 890] """
    def __init__(self, root: str, split: str, num_frames: int, frame_stride: int,
                 image_size: int = 224, transform=None, val_ratio: float = 0.1, seed: int = 42):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.num_frames = num_frames
        self.frame_stride = max(1, frame_stride)
        
        # Tự động đọc class và gom nhóm frames
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        
        grouped_samples = {}
        for cls in self.classes:
            cls_dir = self.root / cls
            for video_dir in sorted([d for d in cls_dir.iterdir() if d.is_dir()]):
                frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
                if not frame_paths: continue
                
                group_key = (cls, self._base_video_name(video_dir.name))
                grouped_samples.setdefault(group_key, []).append((frame_paths, self.class_to_idx[cls]))

        # Chia train/val
        group_values = list(grouped_samples.values())
        rng = np.random.RandomState(seed)
        group_indices = np.arange(len(group_values))
        rng.shuffle(group_indices)
        
        split_point = int(len(group_indices) * (1 - val_ratio))
        if split == "train":
            selected_groups = group_indices[:split_point]
        else:
            selected_groups = group_indices[split_point:]
            
        self.samples = []
        for idx in selected_groups:
            self.samples.extend(group_values[int(idx)])
            
        self.transform = transform or VideoTransform(image_size, is_train=(split == "train"))
        self.to_tensor = transforms.ToTensor()

    def _base_video_name(self, name: str) -> str:
        match = re.match(r"(.+)_\d+$", name)
        return match.group(1) if match else name

    def _select_indices(self, total: int) -> torch.Tensor:
        # Logic sampling và padding
        if total == 0: return torch.zeros(self.num_frames, dtype=torch.long)
        
        steps = max(self.num_frames * self.frame_stride, self.num_frames)
        grid = torch.linspace(0, total - 1, steps=steps)
        idxs = grid[::self.frame_stride].long()
        
        if idxs.numel() < self.num_frames:
            pad = idxs.new_full((self.num_frames - idxs.numel(),), idxs[-1].item())
            idxs = torch.cat([idxs, pad], dim=0)
            
        return idxs[:self.num_frames]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # [cite: 1046]
        frame_paths_list, label = self.samples[idx]
        total = len(frame_paths_list)
        idxs = self._select_indices(total)
        
        frames = []
        for i in idxs:
            path = frame_paths_list[int(i.item())]
            with Image.open(path) as img:
                frames.append(self.to_tensor(img.convert("RGB")))
        
        video = torch.stack(frames) # (T, C, H, W)
        video = self.transform(video)
        
        return video, label

def collate_fn(batch):
    videos = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return videos, labels