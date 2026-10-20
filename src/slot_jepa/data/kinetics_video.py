import os
from pathlib import Path
import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import torchvision.transforms.functional as F_t

import decord
from decord import VideoReader
import torchvision.io as io
import lmdb
import pickle
decord.bridge.set_bridge('torch')


def _canonical_video_name(name: str) -> str:
    name = Path(name).name.strip()
    if name.endswith('.mp4'):
        name = name[:-4]
    return name[2:] if name.startswith('--') else name

class KineticsVideoDataset(Dataset):
    """Kinetics-400 video dataset loader that reads mp4 files from data/Kinetics-400/videos/

    - Expects video folders or mp4 files organized under data/Kinetics-400/videos/{train|val}/
    - Supports returning a clip of `clip_size` frames sampled uniformly within a video.
    - For now, if the project already contains extracted frames, it will fall back to reading frames
      if mp4 reading is not desired. The interface returns a tensor of shape [clip_size, C, H, W]
      when clip_size > 1, or [C,H,W] when clip_size == 1.
    """

    def __init__(self, root: str, split: str = 'train', valid_list: str = '../valid_video_names.txt', clip_size: int = 1, resize: int = 384, transforms: Optional[object] = None, return_info: bool = False):
        root_path = Path(root)
        assert split in ('train', 'val')
        self.split = split
        self.root = root_path / 'videos' / split
        self.clip_size = clip_size
        self.resize = resize
        self.return_info = return_info
        self.transforms = transforms if transforms is not None else T.Compose([T.Resize(resize), T.CenterCrop(resize), T.ToTensor()])

        # collect videos: prefer mp4 files under split directory; else, collect extracted frame folders
        self.items: List[dict] = []
        # whether to try decord when handling mp4s
        self.use_decord = True

        if not self.root.exists():
            raise FileNotFoundError(f"Kinetics videos root not found: {self.root}")


        whitelist_path = root_path / valid_list
        whitelist = set()
        if whitelist_path.exists():
            with open(whitelist_path, 'r') as fh:
                for line in fh:
                    whitelist.add(_canonical_video_name(line))

        for p in sorted((root_path / 'videos' / split).rglob('*.mp4')):
            name = _canonical_video_name(p.stem)
            if (not whitelist) or (name in whitelist):
                self.items.append({'path': p, 'type': 'mp4'})

        # if no mp4s found, search for frame folders
        if len(self.items) == 0:
            self.root = root_path / 'frames' / split # Fix root for frames, typically data/frames/split
            # Directly read from generated list if available
            list_file_path = root_path / f"{split}_frames_list.txt"
            if list_file_path.exists():
                with open(list_file_path, 'r') as list_file:
                    for line in list_file:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            rel_path, num_frames, start_id = parts[0], int(parts[1]), int(parts[2])
                            self.items.append({
                                'path': self.root / rel_path,
                                'type': 'frames',
                                'num_frames': num_frames,
                                'start_id': start_id
                            })

        if len(self.items) == 0:
            raise RuntimeError(f'No videos or frame folders found under {self.root}')

    def __len__(self):
        return len(self.items)

    def _read_frames_from_folder(self, folder: Path, indices: List[int]):
        imgs = sorted([x for x in folder.iterdir() if x.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        frames = []
        for i in indices:
            p = imgs[i]
            frame_tensor = io.read_image(str(p), mode=io.ImageReadMode.RGB)
            frame_tensor = frame_tensor.float() / 255.0
            frames.append(self.transforms(frame_tensor))
        return torch.stack(frames, dim=0)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            orig_idx, target_frame_idx = idx
        else:
            orig_idx, target_frame_idx = idx, None

        current_idx = orig_idx
        for attempt in range(10):
            try:
                item_info = self.items[current_idx]
                item_type = item_info['type']
                item = item_info['path']
                
                if item_type == 'frames':
                    total = item_info['num_frames']
                    start_id = item_info['start_id']
                    def get_img_path(i):
                        return item / f"frame_{start_id + i * 5:05d}.jpg"
                    
                    if self.clip_size == 1:
                        if target_frame_idx is not None:
                            frame_idx = min(target_frame_idx // 5, total - 1)
                        elif self.split == 'train':
                            frame_idx = random.randint(0, total - 1)
                        else:
                            frame_idx = total // 2
                            
                        # Use torchvision.io directly, resulting in [C, H, W] uint8
                        frame_tensor = io.read_image(str(get_img_path(frame_idx)), mode=io.ImageReadMode.RGB)
                        frame_tensor = frame_tensor.float().div_(255.0)
                        out_frame = self.transforms(frame_tensor)
                            
                        if self.return_info:
                            return out_frame, item.name, frame_idx
                        return out_frame
                    else:
                        if total < self.clip_size:
                            indices = [min(i, total-1) for i in range(self.clip_size)]
                        else:
                            if random.random() < 0.7:
                                req_len = 2 * self.clip_size - 1
                                if total < req_len:
                                    start = random.randint(0, max(0, total - self.clip_size))
                                    indices = [min(start + i, total - 1) for i in range(self.clip_size)]
                                else:
                                    start = random.randint(0, total - req_len)
                                    indices = list(range(start, start + req_len, 2))
                            else:
                                window = min(12, total)
                                if window < self.clip_size:
                                    start = random.randint(0, max(0, total - self.clip_size))
                                    indices = [min(start + i, total - 1) for i in range(self.clip_size)]
                                else:
                                    start = random.randint(0, total - window)
                                    window_indices = list(range(start, start + window))
                                    indices = sorted(random.sample(window_indices, self.clip_size))
                        
                        frames = []
                        for idx_i in indices:
                            frames.append(io.read_image(str(get_img_path(idx_i)), mode=io.ImageReadMode.RGB))
                        
                        # zero copy batch stacking & batched division
                        batch_tensor = torch.stack(frames, dim=0).float().div_(255.0)
                        out_frames = self.transforms(batch_tensor)
                        
                        if self.return_info:
                            return out_frames, item.name, indices
                        return out_frames

                if item_type == 'mp4':
                    vr = decord.VideoReader(str(item), num_threads=1)
                    total = len(vr)
                    if total == 0:
                        raise RuntimeError('Empty video')
                    
                    if self.clip_size == 1:
                        if target_frame_idx is not None:
                            frame_idx = min(target_frame_idx, total - 1)
                        elif self.split == 'train':
                            frame_idx = random.randint(0, total - 1)
                        else:
                            frame_idx = total // 2
                            
                        # Since decord.bridge.set_bridge('torch') is set, vr[idx] is a C++ PyTorch backed tensor (H, W, C)
                        frame = vr[frame_idx]
                        frame_tensor = frame.permute(2, 0, 1).float().div_(255.0)
                        out_frame = self.transforms(frame_tensor)
                        del vr
                        
                        if self.return_info:
                            return out_frame, item.stem, frame_idx
                        return out_frame
                    else:
                        if total < self.clip_size:
                            indices = [min(i, total-1) for i in range(self.clip_size)]
                        else:
                            if random.random() < 0.7:
                                req_len = 2 * self.clip_size - 1
                                if total < req_len:
                                    start = random.randint(0, max(0, total - self.clip_size))
                                    indices = [min(start + i, total - 1) for i in range(self.clip_size)]
                                else:
                                    start = random.randint(0, total - req_len)
                                    indices = list(range(start, start + req_len, 2))
                            else:
                                window = min(12, total)
                                if window < self.clip_size:
                                    start = random.randint(0, max(0, total - self.clip_size))
                                    indices = [min(start + i, total - 1) for i in range(self.clip_size)]
                                else:
                                    start = random.randint(0, total - window)
                                    window_indices = list(range(start, start + window))
                                    indices = sorted(random.sample(window_indices, self.clip_size))
                            
                        # Extremely fast parallel decode: vr.get_batch gives [T, H, W, C] natively
                        batch = vr.get_batch(indices)
                        
                        # Permute dimensions [T, H, W, C] -> [T, C, H, W] and perform batched memory division
                        batch_tensor = batch.permute(0, 3, 1, 2).float().div_(255.0)
                        out_frames = self.transforms(batch_tensor)
                        del vr
                        
                        if self.return_info:
                            return out_frames, item.stem, indices
                        return out_frames
            except Exception as e:
                # Handle corrupted video gracefully: draw a completely random new index
                current_idx = random.randint(0, len(self.items) - 1)
                target_frame_idx = None
                
        raise RuntimeError("Failed to decode samples from the KineticsVideoDataset after 10 consecutive fallback retries.")

class KineticsLMDBDataset(Dataset):
    def __init__(self, lmdb_path, split='train', clip_size=1, transforms=None, return_info=False):
        self.lmdb_path = str(lmdb_path)
        self.env = None
        self.txn = None
        self.split = split
        self.clip_size = clip_size
        self.transforms = transforms
        self.return_info = return_info
        
        # 加载 Metadata，得知每个视频有多少帧
        with open(f"{self.lmdb_path}/metadata.pkl", 'rb') as f:
            self.video_metadata = pickle.load(f)

        self.video_names = list(self.video_metadata.keys())
        valid_names_path = Path(self.lmdb_path) / 'valid_video_names.txt'
        if valid_names_path.exists():
            whitelist = set()
            with open(valid_names_path, 'r') as f:
                for line in f:
                    if line.strip():
                        whitelist.add(_canonical_video_name(line))
            self.video_names = [
                video_name for video_name in self.video_names
                if _canonical_video_name(video_name) in whitelist
            ]

    def _init_lmdb(self):
        # 多进程 DataLoader 中，必须在第一个 worker 调用时初始化 env
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path, 
                readonly=True, 
                lock=False,     # 极大提升多进程读取速度
                readahead=False,# 关闭预读，适合随机访问
                meminit=False
            )
            self.txn = self.env.begin(buffers=True)

    def __len__(self):
        return len(self.video_names)

    @staticmethod
    def _decode_jpeg_from_lmdb(img_bytes):
        # LMDB stores sampled frames as JPEG byte streams. Decode directly from JPEG
        # instead of the more generic image decoder.
        encoded = np.frombuffer(memoryview(img_bytes), dtype=np.uint8)
        if not encoded.flags.writeable:
            encoded = encoded.copy()
        byte_tensor = torch.from_numpy(encoded)
        return io.decode_jpeg(byte_tensor, mode=io.ImageReadMode.RGB)

    def __getitem__(self, idx):
        self._init_lmdb()
        
        video_name = self.video_names[idx]
        num_frames = self.video_metadata[video_name]
        
        if self.clip_size == 1:
            target_frame_idx = torch.randint(0, num_frames, (1,)).item() if self.split == 'train' else num_frames // 2
            
            # 构造 Key 并获取数据
            key = f"{video_name}_frame_{target_frame_idx:06d}".encode('ascii')
            img_bytes = self.txn.get(key)
            
            if img_bytes is None:
                return self.__getitem__(torch.randint(0, len(self), (1,)).item())
            
            # 零拷贝读取字节流，并解码为 [C, H, W] 的 uint8 Tensor
            frame_tensor = self._decode_jpeg_from_lmdb(img_bytes)
            
            # 应用针对 uint8 的 transform (切记不要在这里算 div(255) ！)
            if self.transforms:
                frame_tensor = self.transforms(frame_tensor)
                
            if self.return_info:
                return frame_tensor, video_name, target_frame_idx
            return frame_tensor
        else:
            if num_frames < self.clip_size:
                indices = [min(i, num_frames-1) for i in range(self.clip_size)]
            else:
                start_idx = int(torch.randint(0, max(1, num_frames - self.clip_size + 1), (1,)).item())
                indices = [min(start_idx + i, num_frames - 1) for i in range(self.clip_size)]

            frames = []
            for i in indices:
                key = f"{video_name}_frame_{i:06d}".encode('ascii')
                img_bytes = self.txn.get(key)
                if img_bytes is None:
                    return self.__getitem__(torch.randint(0, len(self), (1,)).item())
                frame_tensor = self._decode_jpeg_from_lmdb(img_bytes)
                frames.append(frame_tensor)
                
            batch_tensor = torch.stack(frames, dim=0)
            if self.transforms:
                batch_tensor = self.transforms(batch_tensor)
            if self.return_info:
                return batch_tensor, video_name, indices
            return batch_tensor
