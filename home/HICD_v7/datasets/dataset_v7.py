"""
HICD v7 Dataset - Full image mode (no patch splitting)
Reads full images and resizes to a fixed training size.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    from osgeo import gdal
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False


def read_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.tif', '.tiff'] and HAS_GDAL:
        ds = gdal.Open(path)
        arr = ds.ReadAsArray()
        ds = None
        return arr
    img = Image.open(path).convert('RGB')
    return np.array(img).transpose(2, 0, 1)


def read_label(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.tif', '.tiff'] and HAS_GDAL:
        ds = gdal.Open(path)
        arr = ds.ReadAsArray()
        ds = None
        return arr
    return np.array(Image.open(path))


class FullImageDataset(Dataset):
    """
    Full-image dataset for V7. No patch splitting.
    
    Reads full images, resizes to `img_size`, and provides all
    instance annotations in normalized coordinates.
    """
    
    def __init__(self, data_dir, config, split='train', scenes=None,
                 img_size=1024, boxes_dir=None):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.split = split
        self.img_size = img_size
        self.boxes_dir = boxes_dir
        
        self.pre_suffix = config.get('pre_suffix', '_pre')
        self.post_suffix = config.get('post_suffix', '_post')
        self.label_suffix = config.get('label_suffix', '_label')
        self.image_ext = config.get('image_ext', '.tif')
        self.label_ext = config.get('label_ext', '.tif')
        
        self.label_decode_map = config.get('label_decode_map', None)
        self.is_flat = self._detect_flat_structure()
        
        if scenes is None:
            scenes = self._get_scenes()
        self.scenes = scenes
        
        self.instances_json = self._load_instances_json()
        self.samples = self._collect_samples()
        
        self._img_cache = {}
        self._cache_max = 32
        
        print(f"FullImageDataset: {config.get('dataset', 'unknown')}")
        print(f"  Split: {split}, Scenes: {len(self.scenes)}, Samples: {len(self.samples)}")
        print(f"  Image size: {img_size}x{img_size}")
    
    def _detect_flat_structure(self):
        return os.path.isdir(os.path.join(self.data_dir, self.split, 'image'))
    
    def _get_scenes(self):
        if self.is_flat:
            return ['']
        scenes = []
        for item in sorted(os.listdir(self.data_dir)):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path) and item not in ['train', 'val', 'test']:
                if os.path.exists(os.path.join(item_path, self.split)):
                    scenes.append(item)
        return sorted(scenes)
    
    def _load_instances_json(self):
        json_path = os.path.join(self.data_dir, 'instances.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        return None
    
    def _get_split_dir(self, scene):
        if self.is_flat or scene == '':
            return os.path.join(self.data_dir, self.split)
        return os.path.join(self.data_dir, scene, self.split)
    
    def _collect_samples(self):
        samples = []
        for si, scene in enumerate(self.scenes):
            split_dir = self._get_split_dir(scene)
            img_dir = os.path.join(split_dir, 'image')
            if not os.path.exists(img_dir):
                continue
            lbl_dir = os.path.join(split_dir, 'label')
            
            files = sorted(os.listdir(img_dir))
            for fname in files:
                if not fname.endswith(self.pre_suffix + self.image_ext):
                    continue
                base_name = fname.replace(self.pre_suffix + self.image_ext, '')
                pre_path = os.path.join(img_dir, fname)
                post_path = os.path.join(img_dir, base_name + self.post_suffix + self.image_ext)
                label_path = os.path.join(lbl_dir, base_name + self.label_suffix + self.label_ext)
                
                if not os.path.exists(post_path):
                    continue
                
                inst_key = None
                if self.instances_json is not None:
                    if self.is_flat or scene == '':
                        inst_key = f"{self.split}/{base_name}{self.pre_suffix}{self.image_ext}"
                    else:
                        inst_key = f"{scene}/{self.split}/{base_name}{self.pre_suffix}{self.image_ext}"
                    if inst_key not in self.instances_json:
                        alt_key = f"{self.split}/{base_name}{self.pre_suffix}{self.image_ext}"
                        if alt_key in self.instances_json:
                            inst_key = alt_key
                
                samples.append({
                    'scene': scene,
                    'base_name': base_name,
                    'pre_path': pre_path,
                    'post_path': post_path,
                    'label_path': label_path if os.path.exists(label_path) else None,
                    'inst_key': inst_key,
                })
        
        print(f"  [{self.split}] Total samples: {len(samples)}")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _load_boxes_from_json(self, inst_key):
        if self.instances_json is None or inst_key is None:
            return None, None, None
        entry = self.instances_json.get(inst_key)
        if entry is None:
            return None, None, None
        instances = entry.get('instances', [])
        if not instances:
            return None, None, None
        
        instance_class_names = self.config.get('branch_routing', {}).get('instance', [])
        target_names = self.config.get('target_names', [])
        instance_global_indices = []
        for cls_name in instance_class_names:
            if cls_name in target_names:
                instance_global_indices.append(target_names.index(cls_name))
        
        boxes, classes, states = [], [], []
        for inst in instances:
            bbox = inst.get('bbox')
            target_idx = inst.get('target_idx', 0)
            state_idx = inst.get('state_idx', 0)
            
            if instance_global_indices and target_idx not in instance_global_indices:
                continue
            
            if bbox is not None and len(bbox) == 4:
                boxes.append(bbox)
                if target_idx in instance_global_indices:
                    classes.append(instance_global_indices.index(target_idx))
                else:
                    classes.append(0)
                states.append(state_idx)
        
        if not boxes:
            return None, None, None
        return boxes, classes, states
    
    def _load_boxes_from_txt(self, box_path, img_h, img_w):
        boxes, classes, states = [], [], []
        with open(box_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    # If coordinates are in pixel format, normalize
                    if cx > 1.0 or cy > 1.0:
                        cx /= img_w
                        cy /= img_h
                        w /= img_w
                        h /= img_h
                    boxes.append([cx, cy, w, h])
                    classes.append(cls_id)
                    states.append(int(parts[5]) if len(parts) > 5 else 0)
        if not boxes:
            return None, None, None
        return boxes, classes, states
    
    def _decode_label(self, label):
        if self.label_decode_map is None:
            return label
        decoded = np.zeros_like(label)
        for src, dst in self.label_decode_map.items():
            decoded[label == src] = dst
        return decoded
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        S = self.img_size
        
        # Load images with cache
        cache_key = sample['pre_path']
        if cache_key in self._img_cache:
            pre_img, post_img = self._img_cache[cache_key]
        else:
            pre_img = read_image(sample['pre_path'])
            post_img = read_image(sample['post_path'])
            if len(self._img_cache) < self._cache_max:
                self._img_cache[cache_key] = (pre_img, post_img)
        
        C, H, W = pre_img.shape
        
        # Resize to training size
        pre_pil = Image.fromarray(pre_img.transpose(1, 2, 0).astype(np.uint8))
        post_pil = Image.fromarray(post_img.transpose(1, 2, 0).astype(np.uint8))
        pre_resized = np.array(pre_pil.resize((S, S), Image.BILINEAR)).transpose(2, 0, 1).astype(np.float32) / 255.0
        post_resized = np.array(post_pil.resize((S, S), Image.BILINEAR)).transpose(2, 0, 1).astype(np.float32) / 255.0
        
        result = {
            'img_t1': torch.from_numpy(pre_resized).float(),
            'img_t2': torch.from_numpy(post_resized).float(),
            'scene': sample['scene'],
            'base_name': sample['base_name'],
            'orig_h': H,
            'orig_w': W,
        }
        
        # Load label
        if sample['label_path']:
            try:
                label = read_label(sample['label_path'])
                if len(label.shape) == 3:
                    label = label[0]
                label = self._decode_label(label)
                lbl_pil = Image.fromarray(label.astype(np.uint8))
                lbl_resized = np.array(lbl_pil.resize((S, S), Image.NEAREST))
                result['label'] = torch.from_numpy(lbl_resized).long()
            except Exception:
                pass
        
        # Load instance boxes (normalized coordinates)
        boxes, classes, states = None, None, None
        
        if self.instances_json is not None:
            boxes, classes, states = self._load_boxes_from_json(sample.get('inst_key'))
        
        if boxes is None and self.boxes_dir:
            box_path = os.path.join(
                self.boxes_dir, sample['scene'], self.split, 'boxes',
                sample['base_name'] + self.label_suffix + '.txt'
            )
            if os.path.exists(box_path):
                boxes, classes, states = self._load_boxes_from_txt(box_path, H, W)
        
        if boxes is not None and len(boxes) > 0:
            result['instance_boxes'] = torch.tensor(boxes, dtype=torch.float32)
            result['instance_classes'] = torch.tensor(classes, dtype=torch.long)
            result['instance_states'] = torch.tensor(states, dtype=torch.long)
        
        return result
    
    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return {}
        result = {
            'img_t1': torch.stack([b['img_t1'] for b in batch]),
            'img_t2': torch.stack([b['img_t2'] for b in batch]),
            'scene': [b.get('scene', '') for b in batch],
            'base_name': [b.get('base_name', '') for b in batch],
            'orig_h': [b.get('orig_h', 0) for b in batch],
            'orig_w': [b.get('orig_w', 0) for b in batch],
        }
        if all('label' in b for b in batch):
            result['label'] = torch.stack([b['label'] for b in batch])
        result['instance_boxes'] = [b.get('instance_boxes') for b in batch]
        result['instance_classes'] = [b.get('instance_classes') for b in batch]
        result['instance_states'] = [b.get('instance_states') for b in batch]
        return result


def create_dataloader(data_dir, config, split='train', batch_size=4,
                      num_workers=4, scenes=None, img_size=1024,
                      boxes_dir=None):
    dataset = FullImageDataset(
        data_dir=data_dir, config=config, split=split,
        scenes=scenes, img_size=img_size, boxes_dir=boxes_dir
    )
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers, pin_memory=True,
        drop_last=(split == 'train'),
        collate_fn=FullImageDataset.collate_fn
    )
