"""
Crop images and labels into 256x256 patches.
Also adjusts bounding box coordinates for each patch.
"""

import os
import sys
import numpy as np
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
        if len(arr.shape) == 2:
            arr = np.stack([arr]*3, axis=0)
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


def crop_and_save(img, patch_size, save_func):
    """Crop image into patches and save."""
    C, H, W = img.shape
    patches = []
    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            if y + patch_size <= H and x + patch_size <= W:
                patch = img[:, y:y+patch_size, x:x+patch_size]
                patches.append((y, x, patch))
    return patches


def adjust_boxes(boxes, patch_y, patch_x, patch_size, img_h, img_w):
    """
    Adjust bounding boxes for a cropped patch.
    
    boxes: list of [class_id, cx, cy, w, h] (normalized)
    Returns: list of [cx, cy, w, h] for boxes that fall within the patch
    """
    result = []
    norm_size_h = patch_size / img_h
    norm_size_w = patch_size / img_w
    norm_y = patch_y / img_h
    norm_x = patch_x / img_w
    
    for box in boxes:
        cls_id, cx, cy, w, h = box
        
        # Convert to absolute coords
        abs_cx = cx
        abs_cy = cy
        
        # Check if center is in this patch
        patch_left = norm_x
        patch_right = norm_x + norm_size_w
        patch_top = norm_y
        patch_bottom = norm_y + norm_size_h
        
        if patch_left <= abs_cx < patch_right and patch_top <= abs_cy < patch_bottom:
            # Normalize relative to patch
            new_cx = (abs_cx - patch_left) / norm_size_w
            new_cy = (abs_cy - patch_top) / norm_size_h
            new_w = w / norm_size_w
            new_h = h / norm_size_h
            
            # Clip to [0, 1]
            new_cx = max(0, min(1, new_cx))
            new_cy = max(0, min(1, new_cy))
            new_w = max(0, min(1, new_w))
            new_h = max(0, min(1, new_h))
            
            if new_w > 0.01 and new_h > 0.01:  # Filter tiny boxes
                result.append([new_cx, new_cy, new_w, new_h])
    
    return result


def load_boxes(box_path):
    boxes = []
    with open(box_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return boxes


def process_dataset(data_dir, boxes_dir, output_dir, split='train', patch_size=256):
    """Crop all images, labels, and boxes into patches."""
    
    img_out_dir = os.path.join(output_dir, split, 'image')
    lbl_out_dir = os.path.join(output_dir, split, 'label')
    box_out_dir = os.path.join(output_dir, split, 'boxes')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)
    os.makedirs(box_out_dir, exist_ok=True)
    
    total_patches = 0
    
    for scene in sorted(os.listdir(data_dir)):
        scene_path = os.path.join(data_dir, scene, split)
        img_dir = os.path.join(scene_path, 'image')
        lbl_dir = os.path.join(scene_path, 'label')
        box_dir = os.path.join(boxes_dir, scene, split, 'boxes') if boxes_dir else None
        
        if not os.path.exists(img_dir):
            continue
        
        print(f"  Processing {scene}...")
        
        # Find all pre images
        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith('_pre_war.png'):
                continue
            
            base_name = fname.replace('_pre_war.png', '')
            pre_path = os.path.join(img_dir, fname)
            post_path = os.path.join(img_dir, base_name + '_post_war.png')
            label_path = os.path.join(lbl_dir, base_name + '_target.tif')
            box_path = os.path.join(box_dir, base_name + '_target.txt') if box_dir else None
            
            if not os.path.exists(post_path):
                continue
            
            # Read images
            pre_img = read_image(pre_path)
            post_img = read_image(post_path)
            C, H, W = pre_img.shape
            
            # Read label if exists
            label = None
            if os.path.exists(label_path):
                label = read_label(label_path)
                if len(label.shape) == 3:
                    label = label[0]
            
            # Read boxes if exists
            boxes = None
            if box_path and os.path.exists(box_path):
                boxes = load_boxes(box_path)
            
            # Crop into patches
            patch_idx = 0
            for y in range(0, H, patch_size):
                for x in range(0, W, patch_size):
                    if y + patch_size > H or x + patch_size > W:
                        continue
                    
                    patch_name = f"{base_name}_p{patch_idx:02d}"
                    
                    # Crop and save pre image
                    pre_patch = pre_img[:, y:y+patch_size, x:x+patch_size]
                    Image.fromarray(pre_patch.transpose(1, 2, 0)).save(
                        os.path.join(img_out_dir, patch_name + '_pre_war.png'))
                    
                    # Crop and save post image
                    post_patch = post_img[:, y:y+patch_size, x:x+patch_size]
                    Image.fromarray(post_patch.transpose(1, 2, 0)).save(
                        os.path.join(img_out_dir, patch_name + '_post_war.png'))
                    
                    # Crop and save label
                    if label is not None:
                        lbl_patch = label[y:y+patch_size, x:x+patch_size]
                        Image.fromarray(lbl_patch).save(
                            os.path.join(lbl_out_dir, patch_name + '_target.png'))
                    
                    # Adjust and save boxes
                    if boxes is not None:
                        patch_boxes = adjust_boxes(boxes, y, x, patch_size, H, W)
                        if len(patch_boxes) > 0:
                            with open(os.path.join(box_out_dir, patch_name + '_target.txt'), 'w') as f:
                                for b in patch_boxes:
                                    f.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
                    
                    patch_idx += 1
                    total_patches += 1
    
    print(f"\nTotal patches: {total_patches}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--boxes_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--patch_size', type=int, default=256)
    args = parser.parse_args()
    
    print(f"Cropping {args.split} split into {args.patch_size}x{args.patch_size} patches...")
    process_dataset(args.data_dir, args.boxes_dir, args.output_dir, args.split, args.patch_size)