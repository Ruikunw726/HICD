"""
HICD v6 Dataset Loader - Auto-crop to patches

Supports two directory structures:
  1. Scene-based (XBD): data_dir/scene/split/image/...
  2. Flat (SECOND):     data_dir/split/image/...

Instance boxes can come from:
  - .txt files in boxes_dir (XBD format)
  - instances.json in data_dir (SECOND format)
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


def read_tif(path):
    if HAS_GDAL:
        try:
            ds = gdal.Open(path)
            arr = ds.ReadAsArray()
            ds = None
            return arr
        except Exception:
            pass
    img = Image.open(path)
    arr = np.array(img)
    if len(arr.shape) == 3:
        arr = arr.transpose(2, 0, 1)
    return arr


def read_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.tif', '.tiff']:
        return read_tif(path)
    elif ext in ['.png', '.jpg', '.jpeg']:
        img = Image.open(path).convert('RGB')
        return np.array(img).transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported image format: {ext}")


def read_label(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.tif', '.tiff']:
        return read_tif(path)
    elif ext in ['.png']:
        return np.array(Image.open(path))
    else:
        raise ValueError(f"Unsupported label format: {ext}")


class HICDv6Dataset(Dataset):

    def __init__(self, data_dir, config, split='train', scenes=None,
                 transform=None, boxes_dir=None, patch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.split = split
        self.transform = transform
        self.boxes_dir = boxes_dir
        self.patch_size = patch_size

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
        self.patches = self._collect_patches()
        
        # Image cache (per-worker, since DataLoader forks)
        self._img_cache = {}
        self._cache_max = 64

        print(f"Dataset: {config.get('dataset', 'unknown')}")
        print(f"Structure: {'flat' if self.is_flat else 'scene-based'}")
        print(f"Split: {split}, Scenes: {len(self.scenes)}, Patches: {len(self.patches)}")
        print(f"Patch size: {patch_size}x{patch_size}")
        if self.label_decode_map:
            print(f"Label decode map: {self.label_decode_map}")
        if self.instances_json:
            print(f"Instances JSON loaded: {len(self.instances_json)} entries")

    def _detect_flat_structure(self):
        flat_img = os.path.join(self.data_dir, self.split, 'image')
        return os.path.isdir(flat_img)

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
            print(f"Loading instances.json from {json_path}")
            with open(json_path, 'r') as f:
                return json.load(f)
        return None

    def _get_split_dir(self, scene):
        if self.is_flat or scene == '':
            return os.path.join(self.data_dir, self.split)
        return os.path.join(self.data_dir, scene, self.split)

    def _collect_patches(self):
        import sys
        patches = []
        total_scenes = len(self.scenes)
        for si, scene in enumerate(self.scenes):
            split_dir = self._get_split_dir(scene)
            img_dir = os.path.join(split_dir, 'image')
            if not os.path.exists(img_dir):
                continue
            lbl_dir = os.path.join(split_dir, 'label')

            files = sorted(os.listdir(img_dir))
            print(f"  [{si+1}/{total_scenes}] {scene}: {len(files)} files", flush=True)
            for fname in files:
                if not fname.endswith(self.pre_suffix + self.image_ext):
                    continue

                base_name = fname.replace(self.pre_suffix + self.image_ext, '')
                pre_path = os.path.join(img_dir, fname)
                post_path = os.path.join(img_dir, base_name + self.post_suffix + self.image_ext)
                label_path = os.path.join(lbl_dir, base_name + self.label_suffix + self.label_ext)

                if not os.path.exists(post_path):
                    continue

                try:
                    H, W = self._get_image_size(pre_path)
                except Exception:
                    continue

                inst_key = None
                if self.instances_json is not None:
                    if self.is_flat or scene == '':
                        inst_key = f"{self.split}/{base_name}{self.pre_suffix}{self.image_ext}"
                    else:
                        inst_key = f"{scene}/{self.split}/{base_name}{self.pre_suffix}{self.image_ext}"
                    if inst_key not in self.instances_json:
                        inst_key = f"{self.split}/{base_name}{self.pre_suffix}{self.image_ext}"

                ps = self.patch_size
                for y in range(0, H, ps):
                    for x in range(0, W, ps):
                        if y >= H or x >= W:
                            continue
                        patches.append({
                            'scene': scene,
                            'base_name': base_name,
                            'pre_path': pre_path,
                            'post_path': post_path,
                            'label_path': label_path if os.path.exists(label_path) else None,
                            'patch_y': y,
                            'patch_x': x,
                            'img_h': H,
                            'img_w': W,
                            'inst_key': inst_key,
                        })
        print(f"  Total patches: {len(patches)}", flush=True)
        return patches

    def _get_image_size(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.tif', '.tiff'] and HAS_GDAL:
            ds = gdal.Open(path)
            w = ds.RasterXSize
            h = ds.RasterYSize
            ds = None
            return (h, w)
        img = Image.open(path)
        w, h = img.size
        return (h, w)

    def __len__(self):
        return len(self.patches)

    def _load_boxes_from_txt(self, box_path):
        boxes = []
        classes = []
        states = []
        with open(box_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append([cx, cy, w, h])
                    classes.append(cls_id)
                    states.append(int(parts[5]) if len(parts) > 5 else 0)
        if not boxes:
            return None, None, None
        return boxes, classes, states

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

        boxes = []
        classes = []
        states = []
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

    def _adjust_boxes_for_patch(self, boxes, classes, states, patch_y, patch_x, img_h, img_w):
        ps = self.patch_size
        result_boxes = []
        result_classes = []
        result_states = []
        norm_ps_h = ps / img_h
        norm_ps_w = ps / img_w
        norm_y = patch_y / img_h
        norm_x = patch_x / img_w

        for b, c, s in zip(boxes, classes, states):
            cx, cy, w, h = b
            if norm_x <= cx < norm_x + norm_ps_w and norm_y <= cy < norm_y + norm_ps_h:
                new_cx = (cx - norm_x) / norm_ps_w
                new_cy = (cy - norm_y) / norm_ps_h
                new_w = w / norm_ps_w
                new_h = h / norm_ps_h
                new_cx = max(0.0, min(1.0, new_cx))
                new_cy = max(0.0, min(1.0, new_cy))
                new_w = max(0.01, min(1.0, new_w))
                new_h = max(0.01, min(1.0, new_h))
                result_boxes.append([new_cx, new_cy, new_w, new_h])
                result_classes.append(c)
                result_states.append(s)
        return result_boxes, result_classes, result_states

    def _decode_label(self, label):
        if self.label_decode_map is None:
            return label
        decoded = np.zeros_like(label)
        for src, dst in self.label_decode_map.items():
            decoded[label == src] = dst
        return decoded

    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        ps = self.patch_size
        y, x = patch_info['patch_y'], patch_info['patch_x']

        pre_path = patch_info['pre_path']
        post_path = patch_info['post_path']
        cache_key = pre_path
        
        if cache_key in self._img_cache:
            pre_img, post_img = self._img_cache[cache_key]
        else:
            pre_img = read_image(pre_path)
            post_img = read_image(post_path)
            if len(self._img_cache) < self._cache_max:
                self._img_cache[cache_key] = (pre_img, post_img)
        
        _, img_H, img_W = pre_img.shape

        y_end = min(y + ps, img_H)
        x_end = min(x + ps, img_W)

        pre_patch = pre_img[:, y:y_end, x:x_end].astype(np.float32) / 255.0
        post_patch = post_img[:, y:y_end, x:x_end].astype(np.float32) / 255.0

        if pre_patch.shape[1] < ps or pre_patch.shape[2] < ps:
            pad_h = ps - pre_patch.shape[1]
            pad_w = ps - pre_patch.shape[2]
            pre_patch = np.pad(pre_patch, ((0,0),(0,pad_h),(0,pad_w)), mode='constant')
            post_patch = np.pad(post_patch, ((0,0),(0,pad_h),(0,pad_w)), mode='constant')

        result = {
            'img_t1': torch.from_numpy(pre_patch).float(),
            'img_t2': torch.from_numpy(post_patch).float(),
            'scene': patch_info['scene'],
            'base_name': patch_info['base_name'],
            'patch_y': y,
            'patch_x': x,
            'img_h': patch_info['img_h'],
            'img_w': patch_info['img_w'],
        }

        if patch_info['label_path']:
            try:
                label = read_label(patch_info['label_path'])
                if len(label.shape) == 3:
                    label = label[0]
                label = self._decode_label(label)
                lbl_patch = label[y:y_end, x:x_end]
                if lbl_patch.shape[0] < ps or lbl_patch.shape[1] < ps:
                    pad_h = ps - lbl_patch.shape[0]
                    pad_w = ps - lbl_patch.shape[1]
                    lbl_patch = np.pad(lbl_patch, ((0,pad_h),(0,pad_w)), mode='constant', constant_values=0)
                result['label'] = torch.from_numpy(lbl_patch).long()
            except Exception:
                pass

        boxes, classes, states = None, None, None

        if self.instances_json is not None:
            boxes, classes, states = self._load_boxes_from_json(patch_info.get('inst_key'))

        if boxes is None and self.boxes_dir:
            box_path = os.path.join(
                self.boxes_dir, patch_info['scene'], self.split, 'boxes',
                patch_info['base_name'] + self.label_suffix + '.txt'
            )
            if os.path.exists(box_path):
                boxes, classes, states = self._load_boxes_from_txt(box_path)

        if boxes is not None:
            patch_boxes, patch_classes, patch_states = self._adjust_boxes_for_patch(
                boxes, classes, states, y, x, patch_info['img_h'], patch_info['img_w']
            )
            if patch_boxes:
                result['instance_boxes'] = torch.tensor(patch_boxes, dtype=torch.float32)
                result['instance_classes'] = torch.tensor(patch_classes, dtype=torch.long)
                result['instance_states'] = torch.tensor(patch_states, dtype=torch.long)

        return result

    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return {}
        result = {
            'img_t1': torch.stack([b['img_t1'] for b in batch]),
            'img_t2': torch.stack([b['img_t2'] for b in batch]),
            'scene': [b['scene'] for b in batch],
            'base_name': [b['base_name'] for b in batch],
            'patch_y': [b['patch_y'] for b in batch],
            'patch_x': [b['patch_x'] for b in batch],
            'img_h': [b['img_h'] for b in batch],
            'img_w': [b['img_w'] for b in batch],
        }
        if all('label' in b for b in batch):
            result['label'] = torch.stack([b['label'] for b in batch])
        result['instance_boxes'] = [b.get('instance_boxes') for b in batch]
        result['instance_classes'] = [b.get('instance_classes') for b in batch]
        result['instance_states'] = [b.get('instance_states') for b in batch]
        return result


def create_dataloader(data_dir, config, split='train', batch_size=4,
                      num_workers=4, scenes=None, transform=None,
                      boxes_dir=None, patch_size=256, patch_dir=None):
    # Use explicit patch_dir if provided, else auto-detect {data_dir}_patches/
    use_patch_dir = patch_dir
    if use_patch_dir is None:
        use_patch_dir = data_dir.rstrip('/').rstrip('\\\\') + '_patches'
    if os.path.isdir(use_patch_dir):
        dataset = PrecomputedPatchDataset(
            patch_dir=use_patch_dir, config=config, split=split, scenes=scenes
        )
        collate_fn = PrecomputedPatchDataset.collate_fn
    else:
        dataset = HICDv6Dataset(
            data_dir=data_dir, config=config, split=split, scenes=scenes,
            transform=transform, boxes_dir=boxes_dir, patch_size=patch_size
        )
        collate_fn = HICDv6Dataset.collate_fn
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == 'train'),
        num_workers=num_workers, pin_memory=True, drop_last=(split == 'train'),
        collate_fn=collate_fn
    )

class PrecomputedPatchDataset(Dataset):
    """Fast dataset that reads pre-computed .pt patch files."""

    def __init__(self, patch_dir, config, split='train', scenes=None):
        super().__init__()
        self.patch_dir = patch_dir
        self.config = config
        self.split = split

        # Discover scenes
        if scenes is None:
            scenes = sorted([d for d in os.listdir(patch_dir)
                            if os.path.isdir(os.path.join(patch_dir, d))])
        self.scenes = scenes

        # Collect all .pt files
        self.pt_files = []
        for scene in self.scenes:
            img_dir = os.path.join(patch_dir, scene, split, 'image')
            if not os.path.isdir(img_dir):
                continue
            for f in sorted(os.listdir(img_dir)):
                if f.endswith('.pt'):
                    self.pt_files.append(os.path.join(img_dir, f))

        # Get branch routing
        branch = config.get('branch_routing', {})
        self.instance_class_names = branch.get('instance', [])
        target_names = config.get('target_names', [])
        self.instance_global_indices = []
        for name in self.instance_class_names:
            if name in target_names:
                self.instance_global_indices.append(target_names.index(name))

        print(f"PrecomputedPatchDataset: {split}, {len(self.scenes)} scenes, "
              f"{len(self.pt_files)} patches")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        data = torch.load(self.pt_files[idx], weights_only=False)

        pre = data['pre'].float() / 255.0
        post = data['post'].float() / 255.0

        result = {
            'img_t1': pre,
            'img_t2': post,
            'scene': '',
            'base_name': os.path.basename(self.pt_files[idx]).replace('.pt', ''),
            'patch_y': 0,
            'patch_x': 0,
            'img_h': 0,
            'img_w': 0,
        }

        if 'label' in data:
            result['label'] = data['label'].long()

        # Instance boxes (already filtered to this patch during pre-computation)
        if 'boxes' in data and len(data['boxes']) > 0:
            boxes = data['boxes']
            classes = data.get('classes', torch.zeros(len(boxes), dtype=torch.long))
            states = data.get('states', torch.zeros(len(boxes), dtype=torch.long))

            # Remap global class indices to local instance branch indices
            if self.instance_global_indices:
                local_classes = []
                keep_mask = []
                for i, c in enumerate(classes):
                    c = c.item()
                    if c in self.instance_global_indices:
                        local_classes.append(self.instance_global_indices.index(c))
                        keep_mask.append(i)
                if keep_mask:
                    result['instance_boxes'] = boxes[keep_mask]
                    result['instance_classes'] = torch.tensor(local_classes, dtype=torch.long)
                    result['instance_states'] = states[keep_mask]
                # else: no instance boxes in this patch
            else:
                result['instance_boxes'] = boxes
                result['instance_classes'] = classes
                result['instance_states'] = states

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
            'patch_y': [b.get('patch_y', 0) for b in batch],
            'patch_x': [b.get('patch_x', 0) for b in batch],
            'img_h': [b.get('img_h', 0) for b in batch],
            'img_w': [b.get('img_w', 0) for b in batch],
        }
        if all('label' in b for b in batch):
            result['label'] = torch.stack([b['label'] for b in batch])
        result['instance_boxes'] = [b.get('instance_boxes') for b in batch]
        result['instance_classes'] = [b.get('instance_classes') for b in batch]
        result['instance_states'] = [b.get('instance_states') for b in batch]
        return result
