# -*- coding: utf-8 -*-
"""
像素级标注 → 实例级标注 转换器 (适配 0617final 数据集)

使用 class_mapping.py 统一类别定义, 确保与模型训练一致。

算法:
  1. 读取像素级标注图 (值 = train_id, 0-68)
  2. 通过 class_mapping 将 train_id 分解为 (target_type, state)
  3. 对每种目标类型做连通域分析
  4. 计算 bbox + 多数投票确定目标类型和变化状态
  5. 输出 JSON

用法:
    python pixel_to_instance_0617final.py \
        --data_dir D:\\CD\\0617final\\Airports \\
        --classes_csv D:\\CD\\0617final\\classes.csv \\
        --output D:\\CD\\0617final\\Airports\\instances.json

    # 批量转换所有场景:
    python pixel_to_instance_0617final.py --all_scenes
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
from collections import Counter
from scipy import ndimage

# 导入统一类别定义
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from MambaCD.changedetection.models.class_mapping import (
    TARGET_NAMES, STATE_NAMES, NUM_TARGETS, NUM_STATES,
    TARGET_VALID_STATES, train_id_to_target_state,
)


def load_classes_from_csv(csv_path):
    """
    从 classes.csv 加载映射, 验证与 class_mapping 一致。

    Returns:
        train_id_map: {train_id: (target_idx, state_idx)}
    """
    train_id_map = {}

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_id = int(row['train_id'])
            if train_id == 0:
                continue

            target_en = row['target_en'].strip()
            state = row['state'].strip()

            # 查找 target_idx
            target_idx = None
            for i, name in enumerate(TARGET_NAMES):
                if name.lower() == target_en.lower() or \
                   name.lower() in target_en.lower():
                    target_idx = i
                    break

            # 查找 state_idx
            state_idx = None
            for i, name in enumerate(STATE_NAMES):
                if name.lower() == state.lower() or \
                   name.lower() in state.lower():
                    state_idx = i
                    break

            if target_idx is None:
                print(f"Warning: unknown target '{target_en}' for train_id={train_id}")
                continue
            if state_idx is None:
                print(f"Warning: unknown state '{state}' for train_id={train_id}")
                state_idx = 0

            # 验证有效性
            valid_states = TARGET_VALID_STATES.get(target_idx, [])
            if state_idx not in valid_states:
                print(f"Warning: state {state_idx} ({state}) not valid for "
                      f"target {target_idx} ({TARGET_NAMES[target_idx]})")

            train_id_map[train_id] = (target_idx, state_idx)

    return train_id_map


def connected_components_to_instances(label_map, train_id_map, min_area=50):
    """
    连通域分析, 提取实例。

    对每种目标类型单独做连通域, 同一个连通域内的状态取多数投票。

    Args:
        label_map:      (H, W) 像素级标注图 (train_id 值)
        train_id_map:   {train_id: (target_idx, state_idx)}
        min_area:       最小连通域面积 (像素)

    Returns:
        instances: list of dict
    """
    instances = []
    H, W = label_map.shape

    # 构建目标类型掩码和状态掩码
    target_mask = np.zeros_like(label_map, dtype=np.int32)
    state_mask = np.zeros_like(label_map, dtype=np.int32)

    for train_id, (t_idx, s_idx) in train_id_map.items():
        mask = (label_map == train_id)
        target_mask[mask] = t_idx + 1  # +1 区分背景
        state_mask[mask] = s_idx

    # 对每种目标类型做连通域
    unique_targets = np.unique(target_mask)
    unique_targets = unique_targets[unique_targets > 0]

    for t_val in unique_targets:
        t_idx = t_val - 1
        binary_mask = (target_mask == t_val).astype(np.uint8)
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

            # 归一化 [cx, cy, w, h]
            cx = (x_min + x_max) / 2.0 / W
            cy = (y_min + y_max) / 2.0 / H
            w = (x_max - x_min + 1) / W
            h = (y_max - y_min + 1) / H

            # 多数投票: 目标类型
            inst_targets = target_mask[inst_mask]
            target_vote = int(Counter(inst_targets.tolist()).most_common(1)[0][0]) - 1

            # 多数投票: 变化状态
            inst_states = state_mask[inst_mask]
            state_vote = int(Counter(inst_states.tolist()).most_common(1)[0][0])

            instances.append({
                "bbox": [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)],
                "target_idx": target_vote,
                "state_idx": state_vote,
                "target_name": TARGET_NAMES[target_vote],
                "state_name": STATE_NAMES[state_vote],
                "area": int(area),
            })

    return instances


def process_scene(data_dir, classes_csv, output_path, min_area=50):
    """处理单个场景目录"""
    print(f"\nProcessing: {data_dir}")
    print(f"  Classes CSV: {classes_csv}")

    train_id_map = load_classes_from_csv(classes_csv)
    print(f"  Loaded {len(train_id_map)} class mappings")

    results = {}
    total_instances = 0

    for split in ["train", "val", "test"]:
        label_dir = os.path.join(data_dir, split, "label")
        if not os.path.isdir(label_dir):
            continue

        count = 0
        for label_file in sorted(os.listdir(label_dir)):
            if not (label_file.endswith(".tif") or label_file.endswith(".png")):
                continue

            label_path = os.path.join(label_dir, label_file)

            try:
                from osgeo import gdal
                gdal.UseExceptions()
                ds = gdal.Open(label_path)
                if ds is None:
                    raise ValueError("gdal returned None")
                label_map = ds.ReadAsArray()
                ds = None
            except Exception as e:
                print(f"  Skip {label_path}: {e}")
                continue

            instances = connected_components_to_instances(
                label_map, train_id_map, min_area=min_area
            )

            key = f"{split}/{label_file}"
            results[key] = {
                "image_size": list(label_map.shape),
                "num_instances": len(instances),
                "instances": instances,
            }
            total_instances += len(instances)
            count += 1

        print(f"  {split}: {count} images")

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Total: {len(results)} images, {total_instances} instances")
    print(f"  Saved: {output_path}")

    # 统计
    all_targets = []
    all_states = []
    for v in results.values():
        for inst in v["instances"]:
            all_targets.append(inst['target_idx'])
            all_states.append(inst['state_idx'])

    if all_targets:
        print(f"\n  Target distribution:")
        for t, n in sorted(Counter(all_targets).items()):
            name = TARGET_NAMES[t] if t < len(TARGET_NAMES) else f"?{t}"
            print(f"    [{t:2d}] {name:15s}: {n}")

    if all_states:
        print(f"\n  State distribution:")
        for s, n in sorted(Counter(all_states).items()):
            name = STATE_NAMES[s] if s < len(STATE_NAMES) else f"?{s}"
            print(f"    [{s}] {name:10s}: {n}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert pixel annotations to instance annotations"
    )
    parser.add_argument("--data_dir", type=str,
                        default=r"D:\CD\0617final\Airports",
                        help="场景目录")
    parser.add_argument("--classes_csv", type=str,
                        default=r"D:\CD\0617final\classes.csv")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON 路径 (默认: data_dir/instances.json)")
    parser.add_argument("--min_area", type=int, default=50,
                        help="最小连通域面积")
    parser.add_argument("--all_scenes", action="store_true",
                        help="批量处理所有场景")
    args = parser.parse_args()

    if args.all_scenes:
        base_dir = os.path.dirname(args.data_dir)
        for scene in ["Airports", "Ports", "Urban-Rural Areas"]:
            scene_dir = os.path.join(base_dir, scene)
            if os.path.isdir(scene_dir):
                out = os.path.join(scene_dir, "instances.json")
                process_scene(scene_dir, args.classes_csv, out, args.min_area)
    else:
        output = args.output or os.path.join(args.data_dir, "instances.json")
        process_scene(args.data_dir, args.classes_csv, output, args.min_area)
