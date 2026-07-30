#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量训练脚本 - HICD 层级实例级变化检测
支持: Airports + Ports + Urban-Rural Areas 三个场景联合训练

用法:
    cd /mnt/f/mambacd/home
    export PYTHONPATH="/mnt/f/mambacd/home:$PYTHONPATH"
    source ~/miniconda/bin/activate && conda activate mamba
    
    # 首次运行先准备数据 (在Windows PowerShell中运行 prepare_data.ps1)
    
    # 训练 (预训练权重和CLIP权重已自动加载)
    python HICD/changedetection/script/train_full.py \
        --batch_size 4 \
        --max_epochs 100 \
        --learning_rate 1e-4 \
        --use_amp

"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import sys
import os
import json
import time
import argparse
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from HICD.changedetection.configs.config import get_config
from HICD.changedetection.datasets.imutils import normalize_img
from HICD.changedetection.models.HierarchicalSCD_Instance import HierarchicalSCDInstance
from HICD.changedetection.models.HierarchicalInstanceLoss import HierarchicalInstanceLoss
from HICD.changedetection.models.class_mapping import (
    TARGET_NAMES, STATE_NAMES, NUM_TARGETS, NUM_STATES,
    CLIP_TEXT_PROMPTS,
)

from HICD.changedetection.script.metrics import InstanceMetrics, compute_model_stats

from osgeo import gdal
gdal.UseExceptions()


def win_to_wsl(path):
    """将 Windows 路径转换为 WSL 路径"""
    if path and len(path) >= 2 and path[1] == ':':
        drive = path[0].lower()
        rest = path[2:].replace('\\\\', '/')
        return f'/mnt/{drive}{rest}'
    return path


# =====================================================================
# Dataset (支持扁平目录结构)
# =====================================================================
class ChangeDetectionDataset(Dataset):
    """
    加载双时相影像 + 实例级标注。
    支持两种目录结构:
      1. image/pre/*.tif + image/post/*.tif
      2. image/*_pre_war.tif + image/*_post_war.tif (扁平结构)
    """
    def __init__(self, dataset_path, instances_dict, crop_size=512,
                 max_iters=None, mode="train"):
        self.dataset_path = dataset_path
        self.crop_size = crop_size
        self.instances_dict = instances_dict
        self.mode = mode

        # 检测目录结构
        train_dir = os.path.join(dataset_path, "train", "image")
        self.flat_structure = not os.path.isdir(os.path.join(train_dir, "pre"))

        # 构建样本列表
        self.samples = []
        for key, val in instances_dict.items():
            split, fname = key.split("/", 1)
            if val['num_instances'] > 0:
                self.samples.append((split, fname))

        if max_iters is not None:
            repeats = int(np.ceil(float(max_iters) / len(self.samples)))
            self.samples = self.samples * repeats
            self.samples = self.samples[:max_iters]

        print(f"  Dataset: {len(self.samples)} samples ({mode}, flat={self.flat_structure})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        split, fname = self.samples[index]
        stem = os.path.splitext(fname)[0]
        img_stem = stem.replace('_target', '')  # 图片文件名没有 _target

        # 根据目录结构构建路径
        if self.flat_structure:
            pre_path = os.path.join(self.dataset_path, split, "image", img_stem + "_pre_war.tif")
            post_path = os.path.join(self.dataset_path, split, "image", img_stem + "_post_war.tif")
        else:
            pre_path = os.path.join(self.dataset_path, split, "image", "pre", stem + ".tif")
            post_path = os.path.join(self.dataset_path, split, "image", "post", stem + ".tif")

        # 读取影像 (返回 HWC 格式)
        pre_img = self._read_tif(pre_path).astype(np.float32)
        post_img = self._read_tif(post_path).astype(np.float32)

        # 读取实例标注
        key = f"{split}/{fname}"
        inst_data = self.instances_dict[key]
        instances = inst_data['instances']

        gt_boxes = torch.tensor([inst['bbox'] for inst in instances], dtype=torch.float32)
        gt_target = torch.tensor([inst['target_idx'] for inst in instances], dtype=torch.long)
        gt_state = torch.tensor([inst['state_idx'] for inst in instances], dtype=torch.long)

        # 数据增强
        if self.mode == "train":
            pre_img, post_img, gt_boxes = self._random_augment(pre_img, post_img, gt_boxes)

        # 归一化 (HWC 格式)
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
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3:
            if arr.shape[0] > 3:
                arr = arr[:3]
            arr = np.transpose(arr, (1, 2, 0))  # (C,H,W)->(H,W,C)
        return arr

    def _random_augment(self, pre_img, post_img, gt_boxes):
        """同步随机增强: 翻转 + 旋转"""
        # 水平翻转
        if np.random.rand() > 0.5:
            pre_img = pre_img[:, ::-1, :].copy()
            post_img = post_img[:, ::-1, :].copy()
            if gt_boxes.numel() > 0:
                gt_boxes[:, 0] = 1.0 - gt_boxes[:, 0]

        # 垂直翻转
        if np.random.rand() > 0.5:
            pre_img = pre_img[::-1, :, :].copy()
            post_img = post_img[::-1, :, :].copy()
            if gt_boxes.numel() > 0:
                gt_boxes[:, 1] = 1.0 - gt_boxes[:, 1]

        # 90度旋转 (HWC格式: axes=(0,1))
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
    """自定义 collate: gt_boxes 长度不一"""
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
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # 加载所有场景的 instances 并创建 Dataset
        args.data_dir = win_to_wsl(args.data_dir)
        args.classes_csv = win_to_wsl(args.classes_csv) if args.classes_csv else None
        scenes = args.scenes.split(",")
        print("\nLoading datasets...")
        train_datasets = []
        val_datasets = []

        for scene in scenes:
            scene_dir = os.path.join(args.data_dir, scene.strip())
            json_path = os.path.join(scene_dir, "instances.json")

            if not os.path.exists(json_path):
                print(f"Warning: {json_path} not found, skipping {scene}")
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                instances = json.load(f)

            # 按 train/val 分割 (保留 "train/" 前缀)
            train_inst = {k: v for k, v in instances.items() if k.startswith("train/")}
            val_inst = {k: v for k, v in instances.items() if k.startswith("val/")}

            print(f"  {scene.strip()}: train={len(train_inst)}, val={len(val_inst)}")

            if train_inst:
                train_datasets.append(ChangeDetectionDataset(
                    scene_dir, train_inst, args.crop_size, mode="train"))
            if val_inst:
                val_datasets.append(ChangeDetectionDataset(
                    scene_dir, val_inst, args.crop_size, mode="val"))

        if not train_datasets:
            raise ValueError("No training data found!")

        self.train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        self.val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0] if val_datasets else None

        print(f"\nTotal: train={len(self.train_dataset)}, val={len(self.val_dataset) if self.val_dataset else 0}")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers,
            collate_fn=collate_fn, drop_last=True, pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_fn, pin_memory=True,
        ) if self.val_dataset is not None else None

        # 模型
        print("\nBuilding model...")
        cfg = get_config(args)
        cfg.defrost()
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

        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Model parameters: {total_params:.2f}M")

        # Model complexity (FLOPs)
        model_stats = compute_model_stats(self.model, device=self.device)
        for k, v in model_stats.items():
            print(f'  {k}: {v:.2f}' if isinstance(v, float) else f'  {k}: {v}')

        # Metrics tracker
        self.metrics = InstanceMetrics(
            num_targets=NUM_TARGETS, num_states=NUM_STATES,
            target_names=TARGET_NAMES, state_names=STATE_NAMES,
        )

        # 损失函数
        self.criterion = HierarchicalInstanceLoss(
            num_targets=NUM_TARGETS, num_states=NUM_STATES,
        ).to(self.device)

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # 学习率调度
        # Learning rate: linear warmup (5 epochs) + cosine decay
        total_steps = len(self.train_loader) * args.max_epochs
        warmup_steps = len(self.train_loader) * 5  # 5 epochs warmup
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr = args.learning_rate




        # 混合精度
        self.scaler = torch.amp.GradScaler(enabled=args.use_amp)

        # 保存目录
        self.save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_loss = float('inf')
        self.best_map = 0.0

        # 恢复训练
        self.start_epoch = 0
        if args.resume and os.path.exists(args.resume):
            self._load_checkpoint(args.resume)

    def _load_checkpoint(self, path):
        print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.start_epoch = ckpt.get('epoch', 0)
        self.best_loss = ckpt.get('val_loss', float('inf'))
        self.best_map = ckpt.get('best_map', 0.0)
        print(f"  Epoch: {self.start_epoch}, Best loss: {self.best_loss:.4f}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_start = time.time()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.max_epochs}")

        self.optimizer.zero_grad()
        accum_steps = getattr(self.args, 'grad_accum', 1)
        for batch_idx, batch in enumerate(pbar):
            pre_imgs = batch['pre_img'].to(self.device)
            post_imgs = batch['post_img'].to(self.device)
            gt_boxes_list = [b.to(self.device) for b in batch['gt_boxes']]
            gt_target_list = [t.to(self.device) for t in batch['gt_target']]
            gt_state_list = [s.to(self.device) for s in batch['gt_state']]




            with torch.amp.autocast(device_type='cuda', enabled=self.args.use_amp):
                outputs = self.model(pre_imgs, post_imgs)
                loss, loss_dict = self.criterion(
                    outputs, gt_boxes_list, gt_target_list, gt_state_list
                )

            scaled_loss = loss / accum_steps
            self.scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                if self.args.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step = getattr(self, 'global_step', 0) + 1
                if self.global_step < self.warmup_steps:
                    lr = self.base_lr * self.global_step / self.warmup_steps
                else:
                    progress = (self.global_step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
                    lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr


            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

        epoch_time = time.time() - epoch_start
        self.metrics.train_time += epoch_time
        self.metrics.train_samples += len(self.train_dataset)
        return total_loss / max(num_batches, 1)
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        if self.val_loader is None:
            return float('inf')

        self.metrics.reset()
        total_loss = 0
        num_batches = 0
        infer_start = time.time()

        for batch in tqdm(self.val_loader, desc="Validating"):
            pre_imgs = batch['pre_img'].to(self.device)
            post_imgs = batch['post_img'].to(self.device)
            gt_boxes_list = [b.to(self.device) for b in batch['gt_boxes']]
            gt_target_list = [t.to(self.device) for t in batch['gt_target']]
            gt_state_list = [s.to(self.device) for s in batch['gt_state']]

            with torch.amp.autocast(device_type='cuda', enabled=self.args.use_amp):
                outputs = self.model(pre_imgs, post_imgs)
                loss, _ = self.criterion(
                    outputs, gt_boxes_list, gt_target_list, gt_state_list
                )

            self.metrics.update(outputs, gt_boxes_list, gt_target_list, gt_state_list)
            total_loss += loss.item()
            num_batches += 1

        self.metrics.infer_time = time.time() - infer_start
        self.metrics.infer_samples = len(self.val_dataset)

        val_loss = total_loss / max(num_batches, 1)
        self.val_results = self.metrics.compute()
        self.val_results['val_loss'] = val_loss
        return val_loss
    def train(self):
        print(f"\n{'='*60}")
        print(f"Starting training: {self.args.max_epochs} epochs")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Learning rate: {self.args.learning_rate}")
        print(f"  AMP: {self.args.use_amp}")
        print(f"  Save dir: {self.save_dir}")
        print(f"{'='*60}\n")

        # CSV log
        log_path = os.path.join(self.save_dir, "train_log.csv")
        log_header = "epoch,train_loss,val_loss,mAP@0.5,mAP@0.75,mAP@[0.5:0.95],target_F1,state_F1,train_sps,infer_sps\n"
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write(log_header)

        for epoch in range(self.start_epoch, self.args.max_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            r = self.val_results

            print(f"\nEpoch {epoch+1}/{self.args.max_epochs} — "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            print(self.metrics.format_results(r))

            # Log to CSV
            with open(log_path, 'a') as f:
                f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},"
                        f"{r.get('mAP@0.5', 0):.6f},{r.get('mAP@0.75', 0):.6f},"
                        f"{r.get('mAP@[0.5:0.95]', 0):.6f},"
                        f"{r.get('target_macro_f1', 0):.6f},{r.get('state_macro_f1', 0):.6f},"
                        f"{r.get('train_samples_per_sec', 0):.2f},"
                        f"{r.get('infer_samples_per_sec', 0):.2f}\n")

            # Save best (by mAP@0.5 or val_loss)
            current_map = r.get('mAP@0.5', 0)
            if current_map > self.best_map:
                self.best_map = current_map
                save_path = os.path.join(self.save_dir, "best.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'metrics': {k: float(v) for k, v in r.items() if isinstance(v, (int, float, np.floating))},
                }, save_path)
                print(f"  -> Best model saved (mAP@0.5={current_map:.4f}): {save_path}")

            # Periodic save
            if (epoch + 1) % self.args.save_freq == 0:
                save_path = os.path.join(self.save_dir, f"epoch{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'metrics': {k: float(v) for k, v in r.items() if isinstance(v, (int, float, np.floating))},
                }, save_path)

            # Save latest
            save_path = os.path.join(self.save_dir, "latest.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'best_map': self.best_map,
                'metrics': {k: float(v) for k, v in r.items() if isinstance(v, (int, float, np.floating))},
            }, save_path)

        print(f"\nTraining complete.")
        print(f"  Best val loss: {self.best_loss:.4f}")
        print(f"  Best mAP@0.5: {self.best_map:.4f}")
        print(f"  Log: {log_path}")
        print(f"  Best model: {os.path.join(self.save_dir, 'best.pth')}")

# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HICD Full Training")

    # 数据
    parser.add_argument("--data_dir", type=str, default="HICD/0617final",
                        help="0617final 数据集根目录")
    parser.add_argument("--scenes", type=str, default="Airports,Ports,Urban-Rural Areas",
                        help="训练场景，逗号分隔")
    parser.add_argument("--classes_csv", type=str, default="HICD/0617final/classes.csv")

    # 模型
    parser.add_argument("--pretrained_weight_path", type=str,
                        default="HICD/weights/vssm1_small_0229s_ckpt_epoch_240.pth",
                        help="VSSM 预训练权重路径")
    parser.add_argument("--clip_weights_path", type=str,
                        default="HICD/weights/open_clip_pytorch_model.bin",
                        help="CLIP 权重路径")
    parser.add_argument("--num_queries", type=int, default=17)

    # 训练
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true", help="混合精度训练")
    parser.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数 (等效增大batch_size)")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_freq", type=int, default=10)

    # 输出
    parser.add_argument("--output_dir", type=str, default="HICD/outputs")
    parser.add_argument("--exp_name", type=str, default="full_train")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")

    # VSSM config
    parser.add_argument("--cfg", type=str,
                        default="HICD/changedetection/configs/vssm1/vssm_small_224.yaml")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
