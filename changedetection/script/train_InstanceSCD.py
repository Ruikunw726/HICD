# -*- coding: utf-8 -*-
"""
训练脚本：STMambaSCD_Instance
集成 CLIP 文本引导 + 实例级检测头

用法:
    python train_InstanceSCD.py \
        --dataset test \
        --train_dataset_path /path/to/train \
        --train_data_list_path /path/to/train.txt \
        --test_dataset_path /path/to/test \
        --test_data_list_path /path/to/test.txt \
        --instances_json /path/to/instances.json \
        --pretrained_weight_path /path/to/vssm1_tiny_224.pth \
        --clip_weights_path /path/to/clip_weights \
        --batch_size 4 \
        --crop_size 512 \
        --learning_rate 0.0001 \
        --max_iters 50000
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

from MambaCD.changedetection.configs.config import get_config
from MambaCD.changedetection.datasets.imutils import normalize_img, random_crop_bda, \
    random_fliplr_bda, random_flipud_bda, random_rot_bda
from MambaCD.changedetection.datasets.label_change import airport_add
from MambaCD.changedetection.utils_func.metrics import Evaluator
from MambaCD.changedetection.models.STMambaSCD_Instance import STMambaSCD_Instance
from MambaCD.changedetection.models.InstanceLoss import CombinedLoss
import MambaCD.changedetection.utils_func.lovasz_loss as L
from MambaCD.changedetection.utils_func.mcd_utils import accuracy, SCDD_eval_all, AverageMeter

from osgeo import gdal
gdal.UseExceptions()


# =====================================================================
# Dataset
# =====================================================================
class InstanceChangeDetectionDataset(Dataset):
    """
    同时加载像素级标注和实例级标注。
    
    产出:
        pre_img:        (3, H, W) float32
        post_img:       (3, H, W) float32
        label_cd:       (H, W) long  变化检测标签
        label_clf_t1:   (H, W) long  T1 语义标签
        label_clf_t2:   (H, W) long  T2 语义标签
        gt_boxes:       (M, 4) float32  实例 bbox [cx, cy, w, h]
        gt_labels:      (M,) long       实例损毁类别
    """
    def __init__(self, dataset_path, data_list, crop_size, instances_dict,
                 max_iters=None, mode="train", num_classes=7):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.crop_size = crop_size
        self.instances_dict = instances_dict
        self.mode = mode
        self.num_classes = num_classes
        
        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[:max_iters]
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_idx = self.data_list[index]
        
        # 加载图像
        pre_path = os.path.join(self.dataset_path, "image/pre", data_idx + ".tif")
        post_path = os.path.join(self.dataset_path, "image/post", data_idx + ".tif")
        loc_label_path = os.path.join(self.dataset_path, "label/loc", data_idx + ".tif")
        clf_label_path = os.path.join(self.dataset_path, "label/clf", data_idx + ".tif")
        
        pre_img = self._read_tif(pre_path).astype(np.float32)
        post_img = self._read_tif(post_path).astype(np.float32)
        loc_label = self._read_tif(loc_label_path)
        clf_label = self._read_tif(clf_label_path)
        
        # 数据增强
        if self.mode == "train":
            pre_img, post_img, loc_label, clf_label = random_crop_bda(
                pre_img, post_img, loc_label, clf_label, self.crop_size
            )
            pre_img, post_img, loc_label, clf_label = random_fliplr_bda(
                pre_img, post_img, loc_label, clf_label
            )
            pre_img, post_img, loc_label, clf_label = random_flipud_bda(
                pre_img, post_img, loc_label, clf_label
            )
            pre_img, post_img, loc_label, clf_label = random_rot_bda(
                pre_img, post_img, loc_label, clf_label
            )
        else:
            loc_label = np.asarray(loc_label)
            clf_label = np.asarray(clf_label)
        
        # 标签后处理 (与原代码保持一致)
        clf_label_copy = clf_label.copy()
        clf_label_copy[clf_label_copy == 150] = 2
        clf_label_copy[clf_label_copy == 255] = 1
        clf_label_copy[clf_label_copy == 0] = 255
        
        # 变化检测标签: 有变化=1, 无变化=0
        label_cd = loc_label.copy()
        label_cd[label_cd > 1] = 1
        
        # 语义标签
        label_clf_t1 = clf_label_copy.copy()
        label_clf_t2 = clf_label_copy.copy()
        
        # 归一化图像
        pre_img = normalize_img(pre_img)
        pre_img = np.transpose(pre_img, (2, 0, 1))
        post_img = normalize_img(post_img)
        post_img = np.transpose(post_img, (2, 0, 1))
        
        # 加载实例标注
        gt_boxes, gt_labels = self._load_instances(data_idx, label_cd.shape)
        
        return (
            torch.from_numpy(pre_img).float(),
            torch.from_numpy(post_img).float(),
            torch.from_numpy(label_cd).long(),
            torch.from_numpy(label_clf_t1).long(),
            torch.from_numpy(label_clf_t2).long(),
            torch.from_numpy(gt_boxes).float(),
            torch.from_numpy(gt_labels).long(),
            data_idx,
        )
    
    def _load_instances(self, data_idx, img_shape):
        """从 instances.json 加载实例标注，并根据数据增强同步调整 bbox"""
        # 尝试多个可能的 key 格式
        possible_keys = [
            f"train/{data_idx}.tif",
            f"val/{data_idx}.tif",
            data_idx,
        ]
        
        instances = None
        for key in possible_keys:
            if key in self.instances_dict:
                instances = self.instances_dict[key]["instances"]
                break
        
        if instances is None or len(instances) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)
        
        boxes = []
        labels = []
        for inst in instances:
            boxes.append(inst["bbox"])  # [cx, cy, w, h] 归一化
            labels.append(inst["damage_class"])
        
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    @staticmethod
    def _read_tif(path):
        ds = gdal.Open(path)
        if ds is None:
            raise FileNotFoundError(f"Cannot open: {path}")
        arr = ds.ReadAsArray()
        ds = None
        return arr


def instance_collate_fn(batch):
    """
    自定义 collate：实例标注长度不一，需要 pad。
    """
    pre_imgs, post_imgs, label_cd, label_clf_t1, label_clf_t2, \
        gt_boxes_list, gt_labels_list, data_idxs = zip(*batch)
    
    pre_imgs = torch.stack(pre_imgs, dim=0)
    post_imgs = torch.stack(post_imgs, dim=0)
    label_cd = torch.stack(label_cd, dim=0)
    label_clf_t1 = torch.stack(label_clf_t1, dim=0)
    label_clf_t2 = torch.stack(label_clf_t2, dim=0)
    
    # 实例标注保持 list 形式（每张图实例数不同）
    return pre_imgs, post_imgs, label_cd, label_clf_t1, label_clf_t2, \
        list(gt_boxes_list), list(gt_labels_list), list(data_idxs)


# =====================================================================
# Trainer
# =====================================================================
class InstanceTrainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)
        
        # 加载实例标注
        with open(args.instances_json, "r") as f:
            self.instances_dict = json.load(f)
        
        # 构建数据集
        with open(args.train_data_list_path, "r") as f:
            train_data_name_list = [line.strip() for line in f if line.strip()]
        
        self.train_dataset = InstanceChangeDetectionDataset(
            dataset_path=args.train_dataset_path,
            data_list=train_data_name_list,
            crop_size=args.crop_size,
            instances_dict=self.instances_dict,
            max_iters=args.max_iters,
            mode="train",
            num_classes=7,
        )
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            collate_fn=instance_collate_fn,
        )
        
        # 构建模型
        self.deep_model = STMambaSCD_Instance(
            output_cd=2,
            output_clf=7,
            pretrained=args.pretrained_weight_path,
            num_queries=args.num_queries,
            clip_model="ViT-B-16",
            clip_weights_path=args.clip_weights_path,
            text_prompts=[
                "background, no change",
                "undamaged aircraft",
                "damaged aircraft",
                "undamaged building",
                "damaged building",
                "undamaged vehicle",
                "damaged vehicle",
            ],
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.EMBED_DIM,
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto"
                         else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
        self.deep_model = self.deep_model.cuda()
        
        # 损失函数
        self.criterion = CombinedLoss(
            num_classes=7,
            weight_pixel=1.0,
            weight_instance=1.0,
        )
        
        # 优化器
        self.optim = optim.AdamW(
            self.deep_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optim, step_size=10000, gamma=0.5
        )
        
        # 模型保存路径
        self.model_save_path = os.path.join(
            args.model_param_path, args.dataset,
            f"InstanceSCD_{time.time():.0f}"
        )
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Resume
        if args.resume is not None:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume)
                model_dict = {}
                state_dict = self.deep_model.state_dict()
                for k, v in checkpoint.items():
                    if k in state_dict:
                        model_dict[k] = v
                state_dict.update(model_dict)
                self.deep_model.load_state_dict(state_dict)
                print(f"Resumed from {args.resume}")
    
    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        
        elem_num = len(self.train_data_loader)
        print(f"Training: {elem_num} iterations per epoch")
        
        for epoch in range(args.max_epochs):
            self.deep_model.train()
            train_enumerator = enumerate(self.train_data_loader)
            
            for itera, data in tqdm(train_enumerator, total=elem_num,
                                     desc=f"Epoch {epoch+1}"):
                pre_imgs, post_imgs, label_cd, label_clf_t1, label_clf_t2, \
                    gt_boxes_list, gt_labels_list, _ = data
                
                # 移到 GPU
                pre_imgs = pre_imgs.cuda()
                post_imgs = post_imgs.cuda()
                label_cd = label_cd.cuda()
                label_clf_t1 = label_clf_t1.cuda()
                label_clf_t2 = label_clf_t2.cuda()
                gt_boxes_list = [b.cuda() for b in gt_boxes_list]
                gt_labels_list = [l.cuda() for l in gt_labels_list]
                
                # 前向传播
                outputs = self.deep_model(pre_imgs, post_imgs)
                
                # 计算损失
                loss, loss_dict = self.criterion(
                    outputs, label_cd, label_clf_t1, label_clf_t2,
                    gt_boxes_list, gt_labels_list,
                )
                
                # 加入 Lovasz 损失
                lovasz_cd = L.lovasz_softmax(
                    F.softmax(outputs["pixel_change_map"], dim=1),
                    label_cd, ignore=255
                )
                lovasz_t1 = L.lovasz_softmax(
                    F.softmax(outputs["pixel_T1_map"], dim=1),
                    label_clf_t1, ignore=255
                )
                lovasz_t2 = L.lovasz_softmax(
                    F.softmax(outputs["pixel_T2_map"], dim=1),
                    label_clf_t2, ignore=255
                )
                lovasz_loss = 0.75 * (lovasz_cd + 0.5 * (lovasz_t1 + lovasz_t2))
                
                total_loss = loss + lovasz_loss
                
                # 反向传播
                self.optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.deep_model.parameters(), max_norm=5.0)
                self.optim.step()
                self.scheduler.step()
                
                # 日志
                if (itera + 1) % args.log_freq == 0:
                    print(f"\n[Epoch {epoch+1}][Iter {itera+1}/{elem_num}] "
                          f"Total: {total_loss.item():.4f} | "
                          f"Pixel: {loss_dict['loss_pixel'] if 'loss_pixel' in loss_dict else loss.item():.4f} | "
                          f"Instance: {loss_dict['loss_instance']:.4f} | "
                          f"Change: {loss_dict['loss_change']:.4f} | "
                          f"Lovasz: {lovasz_loss.item():.4f}")
                
                # 保存最优模型
                if (itera + 1) % args.save_freq == 0:
                    torch.save(
                        self.deep_model.state_dict(),
                        os.path.join(self.model_save_path,
                                     f"epoch{epoch+1}_iter{itera+1}.pth")
                    )
        
        print(f"Training finished. Best round: {best_round}")
        print(f"Models saved to: {self.model_save_path}")


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train STMambaSCD_Instance")
    
    # 数据
    parser.add_argument("--dataset", type=str, default="test")
    parser.add_argument("--train_dataset_path", type=str, default=r"E:\code\MambaCD\MambaCD_0326pc\train\train")
    parser.add_argument("--train_data_list_path", type=str, default=r"E:\code\MambaCD\MambaCD_0326pc\train\train.txt")
    parser.add_argument("--test_dataset_path", type=str, default=r"E:\code\MambaCD\MambaCD_0326pc\train\val")
    parser.add_argument("--test_data_list_path", type=str, default=r"E:\code\MambaCD\MambaCD_0326pc\train\val.txt")
    parser.add_argument("--instances_json", type=str, default=r"E:\code\MambaCD\MambaCD_0326pc\train\instances.json")
    
    # 模型权重
    parser.add_argument("--pretrained_weight_path", type=str, default=None)
    parser.add_argument("--clip_weights_path", type=str, 
                        default=r"E:\code\MambaCD\MambaCD_0326pc\changedetection\models\clip_weights")
    parser.add_argument("--resume", type=str, default=None)
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_queries", type=int, default=100)
    
    # 日志与保存
    parser.add_argument("--model_param_path", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=500)
    
    # VSSM config (通过 cfg 文件加载)
    parser.add_argument("--cfg", type=str, 
                        default=r"E:\code\MambaCD\MambaCD_0326pc\changedetection\configs\vssm1\default.yaml")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    
    # 额外参数
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--type", type=str, default="train")
    
    args = parser.parse_args()
    
    # 如果 cfg 文件不存在，设置默认值
    if not os.path.exists(args.cfg):
        print(f"Warning: cfg file not found: {args.cfg}")
        print("Using default VSSM parameters")
    
    trainer = InstanceTrainer(args)
    trainer.training()
