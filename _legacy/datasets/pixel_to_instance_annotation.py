# -*- coding: utf-8 -*-
"""
像素级标注 -> 实例级标注 转换器 (集成 label_change 映射)

算法：
1. 读取像素级标注图
2. 通过 label_change.py 映射为 (loc_label, clf_label)
3. 对每个目标类型做连通域分析
4. 计算 bbox + 多数投票确定损毁类别
5. 输出 JSON

用法:
    python pixel_to_instance_annotation.py --data_dir train --output instances.json
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import Counter
from scipy import ndimage

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from changedetection.datasets.label_change import (
    airport_add, melitopol_airport, test_damage, test_all
)


def read_tif(path):
    """读取 TIF 标注图"""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img
    raise ValueError(f"Cannot read: {path}")


# 根据文件名选择映射函数
MAPPING_FUNCTIONS = {
    "dyagilevo_airport": airport_add,
    "melitopol_airport": melitopol_airport,
}


def get_mapping_func(filename):
    """根据文件名选择对应的标签映射函数"""
    for key, func in MAPPING_FUNCTIONS.items():
        if key in filename:
            return func
    return test_damage  # 默认


def connected_components_to_instances(loc_label, clf_label, min_area=50):
    """
    连通域分析，提取实例。
    
    Args:
        loc_label: (H, W) 目标类型标注（已映射）
        clf_label: (H, W) 损毁等级标注（已映射, 1=未损毁, 2=损毁）
        min_area: 最小连通域面积
    
    Returns:
        instances: list of dict
    """
    instances = []
    H, W = loc_label.shape
    
    unique_classes = np.unique(loc_label)
    unique_classes = unique_classes[(unique_classes != 0) & (unique_classes != 255)]
    
    for cls_val in unique_classes:
        binary_mask = (loc_label == cls_val).astype(np.uint8)
        labeled_array, num_features = ndimage.label(binary_mask)
        
        for inst_id in range(1, num_features + 1):
            inst_mask = (labeled_array == inst_id)
            area = inst_mask.sum()
            
            if area < min_area:
                continue
            
            # 边界框
            ys, xs = np.where(inst_mask)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            
            # 归一化中心点格式
            cx = (x_min + x_max) / 2.0 / W
            cy = (y_min + y_max) / 2.0 / H
            w = (x_max - x_min) / W
            h = (y_max - y_min) / H
            
            # 多数投票损毁类别
            inst_clf = clf_label[inst_mask]
            valid = inst_clf[(inst_clf == 1) | (inst_clf == 2)]
            if len(valid) > 0:
                damage_class = int(Counter(valid.tolist()).most_common(1)[0][0])
            else:
                damage_class = 0
            
            instances.append({
                "bbox": [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)],
                "category_id": int(cls_val),
                "damage_class": damage_class,
                "area": int(area),
            })
    
    return instances


def process_dataset(data_dir, output_path, min_area=50):
    """遍历数据集，转换标注"""
    results = {}
    total_instances = 0
    
    for split in ["train", "val"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        
        for scene_id in sorted(os.listdir(split_dir)):
            scene_dir = os.path.join(split_dir, scene_id)
            if not os.path.isdir(scene_dir):
                continue
            
            label_dir = os.path.join(scene_dir, "label")
            if not os.path.isdir(label_dir):
                continue
            
            for label_file in sorted(os.listdir(label_dir)):
                if not label_file.endswith(".tif"):
                    continue
                
                label_path = os.path.join(label_dir, label_file)
                
                try:
                    raw_label = read_tif(label_path)
                except Exception as e:
                    print(f"  Skip {label_path}: {e}")
                    continue
                
                # 选择映射函数
                map_func = get_mapping_func(label_file)
                
                # 应用标签映射
                loc_label, clf_label = map_func(raw_label.copy(), raw_label.copy())
                
                # 连通域分析
                instances = connected_components_to_instances(
                    loc_label, clf_label, min_area=min_area
                )
                
                key = f"{split}/{scene_id}/{label_file}"
                results[key] = {
                    "image_path": key,
                    "image_size": list(raw_label.shape),
                    "num_instances": len(instances),
                    "instances": instances,
                }
                total_instances += len(instances)
    
    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Done!")
    print(f"  Images: {len(results)}")
    print(f"  Instances: {total_instances}")
    print(f"  Saved: {output_path}")
    
    # 统计
    all_cat = []
    all_dmg = []
    for v in results.values():
        for inst in v["instances"]:
            all_cat.append(inst["category_id"])
            all_dmg.append(inst["damage_class"])
    
    if all_cat:
        print(f"\n  Category distribution:")
        for c, n in sorted(Counter(all_cat).items()):
            print(f"    type_{c}: {n}")
    
    if all_dmg:
        print(f"\n  Damage distribution:")
        for c, n in sorted(Counter(all_dmg).items()):
            label = {0: "unknown", 1: "intact", 2: "damaged"}.get(c, f"class_{c}")
            print(f"    {label}: {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=r"E:\code\HICD\HICD_0326pc\train")
    parser.add_argument("--output", type=str,
                        default=r"E:\code\HICD\HICD_0326pc\train\instances.json")
    parser.add_argument("--min_area", type=int, default=100)
    args = parser.parse_args()
    
    process_dataset(args.data_dir, args.output, args.min_area)
