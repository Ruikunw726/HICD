"""
Pre-extract instance bounding boxes from pixel-level labels.
Run this once before training to avoid slow on-the-fly extraction.

Output format: one .txt file per label, each line is:
  class_id cx cy w h  (normalized coordinates)
"""

import os
import sys
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from osgeo import gdal
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

from PIL import Image


def read_label(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.tif', '.tiff'] and HAS_GDAL:
        ds = gdal.Open(path)
        arr = ds.ReadAsArray()
        ds = None
        return arr
    else:
        return np.array(Image.open(path))


def extract_boxes_from_label(label, min_area=50):
    """
    Extract bounding boxes from pixel-level label.
    
    Args:
        label: [H, W] pixel label (0=background, >0=damage levels)
        min_area: minimum pixel area to be considered an instance
        
    Returns:
        list of [class_id, cx, cy, w, h] (normalized)
    """
    binary_mask = (label > 0).astype(np.uint8)
    
    if binary_mask.sum() == 0:
        return []
    
    labeled, num_instances = ndimage.label(binary_mask)
    
    if num_instances == 0:
        return []
    
    h, w = label.shape
    boxes = []
    
    for i in range(1, num_instances + 1):
        instance_mask = (labeled == i)
        area = instance_mask.sum()
        
        if area < min_area:
            continue
        
        y_coords, x_coords = np.where(instance_mask)
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Get dominant damage level for this instance
        instance_labels = label[instance_mask]
        # Use the most common non-zero label as class
        unique, counts = np.unique(instance_labels, return_counts=True)
        class_id = unique[np.argmax(counts)]
        
        # Normalized (cx, cy, w, h)
        cx = (x_min + x_max) / 2.0 / w
        cy = (y_min + y_max) / 2.0 / h
        bw = (x_max - x_min) / w
        bh = (y_max - y_min) / h
        
        boxes.append([int(class_id), cx, cy, bw, bh])
    
    return boxes


def process_dataset(data_dir, output_dir, split='train'):
    """Process all labels in a dataset split."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_scenes = 0
    total_labels = 0
    total_boxes = 0
    
    for scene in sorted(os.listdir(data_dir)):
        scene_path = os.path.join(data_dir, scene, split)
        label_dir = os.path.join(scene_path, 'label')
        
        if not os.path.exists(label_dir):
            continue
        
        total_scenes += 1
        scene_box_dir = os.path.join(output_dir, scene, split, 'boxes')
        os.makedirs(scene_box_dir, exist_ok=True)
        
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(('.tif', '.tiff', '.png')):
                continue
            
            total_labels += 1
            label_path = os.path.join(label_dir, fname)
            
            try:
                label = read_label(label_path)
                if len(label.shape) == 3:
                    label = label[0]
                
                boxes = extract_boxes_from_label(label)
                total_boxes += len(boxes)
                
                # Save boxes
                box_fname = os.path.splitext(fname)[0] + '.txt'
                box_path = os.path.join(scene_box_dir, box_fname)
                
                with open(box_path, 'w') as f:
                    for box in boxes:
                        f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                
            except Exception as e:
                print(f"  Error processing {label_path}: {e}")
        
        print(f"  {scene}: processed labels in {split}/")
    
    print(f"\nSummary:")
    print(f"  Scenes: {total_scenes}")
    print(f"  Labels processed: {total_labels}")
    print(f"  Total boxes extracted: {total_boxes}")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract instance bounding boxes from pixel labels')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for boxes')
    parser.add_argument('--splits', type=str, default='train', help='Comma-separated splits to process')
    
    args = parser.parse_args()
    
    for split in args.splits.split(','):
        print(f"\nProcessing {split} split...")
        process_dataset(args.data_dir, args.output_dir, split.strip())
