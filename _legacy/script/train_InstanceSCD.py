# -*- coding: utf-8 -*-
"""
训练脚本: HierarchicalSCDInstance (层级实例级语义变化检测)

用法:
    python train_InstanceSCD.py \
        --data_dir D:/CD/0617final/Airports \
        --classes_csv D:\\CD\\0617final\\classes.csv \
        --pretrained_weight_path /path/to/vssm1_tiny_224.pth \
        --clip_weights_path /path/to/clip_weights \
        --batch_size 2 \
        --crop_size 512 \
        --max_epochs 100

数据目录结构:
    data_dir/
    ├── train/
    │   ├── image/pre/*.tif
    │   ├── image/post/*.tif
    │   └── label/*.tif
    ├── val/
    │   └── ...
    └── instances.json  (由 pixel_to_instance_0617final.py 生成)
"""

import sys
import os
import argparse
import time
import json
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from HICD.changedetection.configs.config import get_config
from HICD.changedetection.datasets.imutils import (
    normalize_img, random_crop_bda, random_fliplr_bda,
    random_flipud_bda, random_rot_bda,
)
from HICD.changedetection.models.HierarchicalSCD_Instance import HierarchicalSCDInstance
from HICD.changedetection.models.HierarchicalInstanceLoss import HierarchicalInstanceLoss
from HICD.changedetection.models.class_mapping import (
    TARGET_NAMES, STATE_NAMES, NUM_TARGETS, NUM_STATES,
    train_id_to_target_state,
)

from osgeo import gdal
gdal.UseExceptions()


# =====================================================================
# Dataset
# =====================================================================
class InstanceChangeDetectionDataset(Dataset):
    """
    加载双时相影像 + 实例级标注。

    产出:
        pre_img:      (3, H, W) float32
        post_img:     (3, H, W) float32
        gt_boxes:     (M, 4) float32  [cx, cy, w, h] 归一化
        gt_target:    (M,) long       目标类型 0-15
        gt_state:     (M,) long       变化状态 0-5
    """
    def __init__(self, dataset_path, instances_dict, crop_size,
                 max_iters=None, mode="train"):
        self.dataset_path = dataset_path
        self.crop_size = crop_size
        self.instances_dict = instances_dict
        self.mode = mode

        # 构建样本列表: [(split, filename), ...]
        self.samples = []
        for key, val in instances_dict.items():
            split, fname = key.split("/", 1)
            # 只加载有实例的样本
            if val['num_instances'] > 0:
                self.samples.append((split, fname))

        if max_iters is not None:
            repeats = int(np.ceil(float(max_iters) / len(self.samples)))
            self.samples = self.samples * repeats
            self.samples = self.samples[:max_iters]

        print(f"Dataset: {len(self.samples)} samples "
              f"({mode}, {dataset_path})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        split, fname = self.samples[index]
        stem = os.path.splitext(fname)[0]

        # 路径
        pre_path = os.path.join(self.dataset_path, split, "image", "pre", stem + ".tif")
        post_path = os.path.join(self.dataset_path, split, "image", "post", stem + ".tif")

        # 读取影像
        pre_img = self._read_tif(pre_path).astype(np.float32)
        post_img = self._read_tif(post_path).astype(np.float32)

        # 读取实例标注
        key = f"{split}/{fname}"
        inst_data = self.instances_dict[key]
        instances = inst_data['instances']
        img_h, img_w = inst_data['image_size']

        # 转换为 tensor
        gt_boxes = torch.tensor(
            [inst['bbox'] for inst in instances], dtype=torch.float32
        )
        gt_target = torch.tensor(
            [inst['target_idx'] for inst in instances], dtype=torch.long
        )
        gt_state = torch.tensor(
            [inst['state_idx'] for inst in instances], dtype=torch.long
        )

        # 数据增强 (训练时)
        if self.mode == "train":
            pre_img, post_img, gt_boxes = self._random_augment(
                pre_img, post_img, gt_boxes
            )

        # 归一化
        pre_img = normalize_img(pre_img)
        post_img = normalize_img(post_img)

        # 转为 CHW
        pre_img = np.transpose(pre_img, (2, 0, 1)).astype(np.float32)
        post_img = np.transpose(post_img, (2, 0, 1)).astype(np.float32)

        return {
            'pre_img': torch.from_numpy(pre_img),
            'post_img': torch.from_numpy(post_img),
            'gt_boxes': gt_boxes,
            'gt_target': gt_target,
            'gt_state': gt_state,
            'filename': key,
        }

    def _read_tif(self, path):
        ds = gdal.Open(path)
        if ds is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        arr = ds.ReadAsArray()
        ds = None
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)  # (H,W,3)
        elif arr.ndim == 3:
            if arr.shape[0] > 3:
                arr = arr[:3]  # 取前3个波段
            arr = np.transpose(arr, (1, 2, 0))  # (C,H,W)->(H,W,C)
        return arr

    def _random_augment(self, pre_img, post_img, gt_boxes):
        """同步随机增强: 翻转 + 旋转, bbox 同步变换"""
        H, W = pre_img.shape[:2]

        # 随机水平翻转
        if np.random.rand() > 0.5:
            pre_img = pre_img[:, ::-1, :].copy()
            post_img = post_img[:, ::-1, :].copy()
            if gt_boxes.numel() > 0:
                gt_boxes[:, 0] = 1.0 - gt_boxes[:, 0]

        # 随机垂直翻转
        if np.random.rand() > 0.5:
            pre_img = pre_img[::-1, :, :].copy()
            post_img = post_img[::-1, :, :].copy()
            if gt_boxes.numel() > 0:
                gt_boxes[:, 1] = 1.0 - gt_boxes[:, 1]

        # 随机 90° 旋转 (HWC格式: axes=(0,1)旋转H-W平面)
        k = np.random.randint(0, 4)
        if k > 0:
            pre_img = np.rot90(pre_img, k, axes=(0, 1)).copy()
            post_img = np.rot90(post_img, k, axes=(0, 1)).copy()
            if gt_boxes.numel() > 0:
                for _ in range(k):
                    cx, cy, w, h = gt_boxes.unbind(-1)
                    gt_boxes = torch.stack([cy, cx, h, w], dim=-1)
                    gt_boxes[:, 0] = 1.0 - gt_boxes[:, 0]

        return pre_img, post_img, gt_boxes

def collate_fn(batch):
    """自定义 collate: gt_boxes/gt_target/gt_state 长度不一, 用 list 传递"""
    return {
        'pre_img': torch.stack([b['pre_img'] for b in batch]),
        'post_img': torch.stack([b['post_img'] for b in batch]),
        'gt_boxes': [b['gt_boxes'] for b in batch],
        'gt_target': [b['gt_target'] for b in batch],
        'gt_state': [b['gt_state'] for b in batch],
        'filename': [b['filename'] for b in batch],
    }


# =====================================================================
# Trainer
# =====================================================================
class InstanceTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载实例标注
        print(f"Loading instances from: {args.instances_json}")
        with open(args.instances_json, 'r', encoding='utf-8') as f:
            self.all_instances = json.load(f)

        # 分割 train/val
        self.train_instances = {
            k: v for k, v in self.all_instances.items()
            if k.startswith("train/")
        }
        self.val_instances = {
            k: v for k, v in self.all_instances.items()
            if k.startswith("val/")
        }
        # 如果没有 val, 用 train 的一部分
        if len(self.val_instances) == 0:
            keys = list(self.train_instances.keys())
            split_idx = int(len(keys) * 0.9)
            self.val_instances = {k: self.train_instances[k] for k in keys[split_idx:]}
            self.train_instances = {k: self.train_instances[k] for k in keys[:split_idx]}

        print(f"  Train: {len(self.train_instances)} images")
        print(f"  Val:   {len(self.val_instances)} images")

        # Dataset
        self.train_dataset = InstanceChangeDetectionDataset(
            args.data_dir, self.train_instances, args.crop_size,
            max_iters=args.max_iters, mode="train"
        )
        self.val_dataset = InstanceChangeDetectionDataset(
            args.data_dir, self.val_instances, args.crop_size,
            mode="val"
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers,
            collate_fn=collate_fn, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

        # Model
        print("Building model...")
        cfg = get_config(args)
        cfg.defrost()
        # Build complete kwargs dict from config
        vssm = cfg.MODEL.VSSM
        cfg_dict = {
            'norm_layer': vssm.NORM_LAYER,
            'ssm_act_layer': vssm.SSM_ACT_LAYER,
            'mlp_act_layer': vssm.MLP_ACT_LAYER,
            'ssm_d_state': vssm.SSM_D_STATE,
            'ssm_ratio': vssm.SSM_RATIO,
            'ssm_dt_rank': vssm.SSM_DT_RANK,
            'ssm_conv': vssm.SSM_CONV,
            'ssm_conv_bias': vssm.SSM_CONV_BIAS,
            'ssm_drop_rate': vssm.SSM_DROP_RATE,
            'ssm_init': vssm.SSM_INIT,
            'forward_type': vssm.SSM_FORWARDTYPE,
            'mlp_ratio': vssm.MLP_RATIO,
            'mlp_drop_rate': vssm.MLP_DROP_RATE,
            'gmlp': vssm.GMLP,
            'use_checkpoint': cfg.TRAIN.USE_CHECKPOINT,
            'drop_path_rate': cfg.MODEL.DROP_PATH_RATE,
            'patch_size': vssm.PATCH_SIZE,
            'in_chans': vssm.IN_CHANS,
            'embed_dim': vssm.EMBED_DIM,
            'depths': vssm.DEPTHS,
            'downsample': vssm.DOWNSAMPLE,
            'patchembed': vssm.PATCHEMBED,
            'patch_norm': vssm.PATCH_NORM,
        }
        self.model = HierarchicalSCDInstance(
            pretrained=args.pretrained_weight_path,
            num_queries_per_scale=args.num_queries,
            clip_weights_path=args.clip_weights_path,
            **cfg_dict,
        ).to(self.device)

        # Loss
        self.criterion = HierarchicalInstanceLoss(
            num_targets=NUM_TARGETS, num_states=NUM_STATES,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # Scheduler
        total_steps = len(self.train_loader) * args.max_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-7
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler(enabled=args.use_amp)

        # Checkpoint
        self.model_save_path = os.path.join(args.data_dir, "checkpoints_instance")
        os.makedirs(self.model_save_path, exist_ok=True)

        self.best_loss = float('inf')
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            pre_imgs = batch['pre_img'].to(self.device)
            post_imgs = batch['post_img'].to(self.device)
            gt_boxes_list = batch['gt_boxes']
            gt_target_list = batch['gt_target']
            gt_state_list = batch['gt_state']

            # 移到 GPU
            gt_boxes_list = [b.to(self.device) for b in gt_boxes_list]
            gt_target_list = [t.to(self.device) for t in gt_target_list]
            gt_state_list = [s.to(self.device) for s in gt_state_list]

            with torch.amp.autocast('cuda', enabled=self.args.use_amp):
                outputs = self.model(pre_imgs, post_imgs)
                loss, loss_dict = self.criterion(
                    outputs, gt_boxes_list, gt_target_list, gt_state_list
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=5.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bbox': f"{loss_dict['loss_bbox']:.4f}",
                'target': f"{loss_dict['loss_target']:.4f}",
                'state': f"{loss_dict['loss_state']:.4f}",
            })

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            pre_imgs = batch['pre_img'].to(self.device)
            post_imgs = batch['post_img'].to(self.device)
            gt_boxes_list = [b.to(self.device) for b in batch['gt_boxes']]
            gt_target_list = [t.to(self.device) for t in batch['gt_target']]
            gt_state_list = [s.to(self.device) for s in batch['gt_state']]

            outputs = self.model(pre_imgs, post_imgs)
            loss, _ = self.criterion(
                outputs, gt_boxes_list, gt_target_list, gt_state_list
            )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(self):
        print(f"\nStarting training for {self.args.max_epochs} epochs")
        print(f"  Device: {self.device}")
        print(f"  AMP: {self.args.use_amp}")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  LR: {self.args.learning_rate}")
        print(f"  Queries per scale: {self.args.num_queries}")
        print(f"  Total queries: {3 * self.args.num_queries}")
        print()

        for epoch in range(self.args.max_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            print(f"\nEpoch {epoch+1}/{self.args.max_epochs} — "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # 保存最优
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                save_path = os.path.join(self.model_save_path, "best.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
                print(f"  → Best model saved: {save_path}")

            # 定期保存
            if (epoch + 1) % self.args.save_freq == 0:
                save_path = os.path.join(
                    self.model_save_path,
                    f"epoch{epoch+1}.pth"
                )
                torch.save(self.model.state_dict(), save_path)

        print(f"\nTraining complete. Best val loss: {self.best_loss:.4f}")


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train HierarchicalSCDInstance"
    )

    # 数据
    parser.add_argument("--data_dir", type=str,
                        default=r"D:\CD\0617final\Airports",
                        help="场景目录 (包含 train/val/image/label)")
    parser.add_argument("--classes_csv", type=str,
                        default=r"D:\CD\0617final\classes.csv")
    parser.add_argument("--instances_json", type=str,
                        default=r"D:\CD\0617final\Airports\instances.json",
                        help="实例标注 JSON")

    # 模型
    parser.add_argument("--pretrained_weight_path", type=str, default=None,
                        help="VSSM 预训练权重")
    parser.add_argument("--clip_weights_path", type=str, default=None,
                        help="CLIP 权重路径")
    parser.add_argument("--num_queries", type=int, default=34,
                        help="每个尺度的查询数")

    # 训练
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=None,
                        help="每 epoch 迭代次数 (None=全部)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true",
                        help="混合精度训练")
    parser.add_argument("--save_freq", type=int, default=10,
                        help="每 N epoch 保存一次")

    # VSSM config
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)

    args = parser.parse_args()

    # 检查文件
    if not os.path.exists(args.instances_json):
        print(f"Error: instances.json not found: {args.instances_json}")
        print("Run pixel_to_instance_0617final.py first.")
        sys.exit(1)

    trainer = InstanceTrainer(args)
    trainer.train()
