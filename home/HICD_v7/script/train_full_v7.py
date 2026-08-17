"""
HICD v7 Training Script
Full-image training, dual-branch with interaction layer, proper logging.
"""

import os
import sys
import argparse
import yaml
import time
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_hicd_v7
from datasets import create_dataloader
from evaluation import ICDEvaluatorV6 as ICDEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='HICD v7 Training')
    
    # Dataset
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--scenes', type=str, default=None)
    parser.add_argument('--boxes_dir', type=str, default=None)
    parser.add_argument('--config_dir', type=str, default='datasets/configs')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='Training image size (full image, no cropping)')
    
    # Model
    parser.add_argument('--backbone', type=str, default='L1')
    parser.add_argument('--pretrained_weight_path', type=str, default=None)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_queries', type=int, default=100)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--freeze_backbone_epochs', type=int, default=10)
    parser.add_argument('--unfreeze_ratio', type=float, default=0.5)
    
    # Loss weights
    parser.add_argument('--w_instance', type=float, default=1.0)
    parser.add_argument('--w_pixel', type=float, default=1.0)
    parser.add_argument('--w_change_type', type=float, default=1.0)
    parser.add_argument('--w_damage_level', type=float, default=1.0)
    parser.add_argument('--w_consistency', type=float, default=0.3)
    
    # Saving
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--exp_name', type=str, default='v7_xbd')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=50)
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


class TrainerV7:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load dataset config
        config_path = os.path.join(args.config_dir, f'{args.dataset}.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.dataset_config = yaml.safe_load(f)
        
        print(f"Dataset config: {self.dataset_config}")
        
        # Build model
        model_config = self._build_model_config()
        self.model = build_hicd_v7(model_config).to(self.device)
        self._print_model_summary()
        
        # Dataloaders
        scenes = args.scenes.split(',') if args.scenes else None
        self.train_loader = create_dataloader(
            data_dir=args.data_dir, config=self.dataset_config,
            split='train', batch_size=args.batch_size,
            num_workers=args.num_workers, scenes=scenes,
            img_size=args.img_size, boxes_dir=args.boxes_dir
        )
        self.val_loader = create_dataloader(
            data_dir=args.data_dir, config=self.dataset_config,
            split='val', batch_size=args.batch_size,
            num_workers=args.num_workers, scenes=scenes,
            img_size=args.img_size, boxes_dir=args.boxes_dir
        )
        
        # Optimizer
        param_groups = self.model.get_param_groups(
            lr=args.learning_rate, weight_decay=args.weight_decay
        )
        self.optimizer = optim.AdamW(param_groups)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.max_epochs,
            eta_min=args.learning_rate * 0.01
        )
        
        # AMP
        self.scaler = GradScaler('cuda') if args.use_amp else None
        
        # Evaluator
        self.evaluator = ICDEvaluator(self.dataset_config)
        
        # Save dir
        self.save_dir = os.path.join(args.save_dir, args.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # CSV log
        self.csv_path = os.path.join(self.save_dir, 'training_log.csv')
        self._init_csv()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Resume
        if args.resume and os.path.exists(args.resume):
            self._load_checkpoint(args.resume)
    
    def _build_model_config(self):
        config = {
            'backbone': {
                'variant': self.args.backbone,
                'pretrained': self.args.pretrained_weight_path is not None,
                'pretrained_path': self.args.pretrained_weight_path
            },
            'branch_routing': self.dataset_config.get('branch_routing', {}),
            'clip': {'enabled': False, 'model_path': None},
            'active_heads': self.dataset_config.get('active_heads', {}),
            'num_targets': self.dataset_config.get('num_targets', 1),
            'num_pixel_targets': self.dataset_config.get('num_pixel_targets', self.dataset_config.get('num_targets', 1)),
            'num_states': self.dataset_config.get('num_states', 5),
            'num_change_types': self.dataset_config.get('num_change_types', 5),
            'num_damage_levels': self.dataset_config.get('num_damage_levels', 4),
            'hidden_dim': self.args.hidden_dim,
            'num_queries': self.args.num_queries,
            'nhead': self.args.nhead,
            'num_decoder_layers': self.args.num_decoder_layers,
            'w_instance': self.args.w_instance,
            'w_pixel': self.args.w_pixel,
            'w_change_type': self.args.w_change_type,
            'w_damage_level': self.args.w_damage_level,
            'w_consistency': self.args.w_consistency,
            'state_names': self.dataset_config.get('state_names', []),
        }
        return config
    
    def _print_model_summary(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"Model Summary")
        print(f"{'='*60}")
        print(f"Total parameters: {total:,} ({total/1e6:.2f}M)")
        print(f"Trainable parameters: {trainable:,} ({trainable/1e6:.2f}M)")
        print(f"{'='*60}\n")
    
    def _init_csv(self):
        header = [
            'epoch', 'train_loss', 'val_loss',
            'mAP@0.5', 'mAP@0.75', 'mAP@[.5:.95]',
            'inst_loss_class', 'inst_loss_bbox', 'inst_loss_giou',
            'inst_loss_state', 'inst_total',
            'loss_change_type', 'loss_damage_level', 'loss_consistency',
            'loss_pixel', 'total'
        ]
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(header)
    
    def _log_to_csv(self, epoch, train_loss, val_loss, eval_results, loss_dict):
        row = [
            epoch + 1,
            f"{train_loss:.4f}",
            f"{val_loss:.4f}",
            f"{eval_results.get('instance_mAP@0.5', 0):.4f}",
            f"{eval_results.get('instance_mAP@0.75', 0):.4f}",
            f"{eval_results.get('instance_mAP@[0.5:0.95]', 0):.4f}",
            f"{loss_dict.get('inst_loss_class', 0):.4f}",
            f"{loss_dict.get('inst_loss_bbox', 0):.4f}",
            f"{loss_dict.get('inst_loss_giou', 0):.4f}",
            f"{loss_dict.get('inst_loss_state', 0):.4f}",
            f"{loss_dict.get('inst_total', 0):.4f}",
            f"{loss_dict.get('loss_change_type', 0):.4f}",
            f"{loss_dict.get('loss_damage_level', 0):.4f}",
            f"{loss_dict.get('loss_consistency', 0):.4f}",
            f"{loss_dict.get('loss_pixel', 0):.4f}",
            f"{loss_dict.get('total', 0):.4f}",
        ]
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
    
    def _prepare_targets(self, batch):
        targets = {}
        
        if 'instance_boxes' in batch:
            instance_labels = []
            instance_classes = batch.get('instance_classes')
            instance_states = batch.get('instance_states')
            for i, boxes in enumerate(batch['instance_boxes']):
                if boxes is not None and len(boxes) > 0:
                    labels = (instance_classes[i].to(self.device) 
                              if instance_classes is not None and instance_classes[i] is not None
                              else torch.zeros(len(boxes), dtype=torch.long, device=self.device))
                    inst_dict = {'labels': labels, 'boxes': boxes.to(self.device)}
                    if instance_states is not None and instance_states[i] is not None:
                        inst_dict['states'] = instance_states[i].to(self.device)
                    instance_labels.append(inst_dict)
                else:
                    instance_labels.append({
                        'labels': torch.zeros(0, dtype=torch.long, device=self.device),
                        'boxes': torch.zeros(0, 4, device=self.device)
                    })
            targets['instance_labels'] = instance_labels
        
        # Pixel-level targets
        semantic_classes = self.dataset_config.get('branch_routing', {}).get('semantic', [])
        if 'label' in batch and len(semantic_classes) > 0:
            label = batch['label'].to(self.device)
            if label.dim() == 3:
                label_s = label.unsqueeze(1).float()
                label_s = nn.functional.interpolate(
                    label_s, scale_factor=0.25, mode='nearest'
                ).squeeze(1).long()
                targets['target_gt'] = label_s
        
        # Damage level targets (from instance_states)
        if 'instance_states' in batch:
            all_states = []
            for i, states in enumerate(batch['instance_states']):
                if states is not None:
                    all_states.append(states.to(self.device))
                else:
                    all_states.append(torch.zeros(0, dtype=torch.long, device=self.device))
            targets['damage_level_gt_batch'] = all_states
        
        return targets
    
    def train(self):
        print(f"\nStarting training for {self.args.max_epochs} epochs")
        print(f"  Device: {self.device}")
        print(f"  AMP: {self.args.use_amp}")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Image size: {self.args.img_size}")
        print(f"  CSV log: {self.csv_path}")
        
        for epoch in range(self.current_epoch, self.args.max_epochs):
            self.current_epoch = epoch
            
            # Backbone freeze strategy
            if epoch < self.args.freeze_backbone_epochs:
                if epoch == 0:
                    print(f"\nStage 1: Freezing backbone for {self.args.freeze_backbone_epochs} epochs")
                    self.model.freeze_backbone()
            elif epoch == self.args.freeze_backbone_epochs:
                print(f"\nStage 2: Unfreezing backbone (ratio={self.args.unfreeze_ratio})")
                self.model.unfreeze_backbone(self.args.unfreeze_ratio)
            elif epoch == self.args.max_epochs // 2:
                print(f"\nStage 3: Unfreezing entire backbone")
                self.model.unfreeze_backbone(1.0)
            
            # Train
            train_loss, train_loss_dict = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_results = self.validate(epoch)
            
            # LR scheduler
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch, train_loss)
            
            # Save best
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print(f"  ** New best model **")
            
            # Log to CSV
            self._log_to_csv(epoch, train_loss, val_loss, val_results, train_loss_dict)
            
            print(f'Epoch {epoch+1}/{self.args.max_epochs} '
                  f'- Train Loss: {train_loss:.4f} '
                  f'- Val Loss: {val_loss:.4f} '
                  f'- mAP@0.5: {val_results.get("instance_mAP@0.5", 0):.4f} '
                  f'- Best: {self.best_loss:.4f}')
        
        print(f"\nTraining completed! Best loss: {self.best_loss:.4f}")
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        loss_accum = {}
        
        self.evaluator.reset()
        
        pbar = tqdm(self.train_loader,
                    desc=f'Epoch {epoch+1}/{self.args.max_epochs}',
                    leave=True, ncols=120, mininterval=0.5, file=sys.stdout)
        
        for batch_idx, batch in enumerate(pbar):
            img_t1 = batch['img_t1'].to(self.device)
            img_t2 = batch['img_t2'].to(self.device)
            targets = self._prepare_targets(batch)
            
            if self.args.use_amp:
                with autocast('cuda'):
                    predictions, loss, loss_dict = self.model(img_t1, img_t2, targets)
            else:
                predictions, loss, loss_dict = self.model(img_t1, img_t2, targets)
            
            scaled_loss = loss / self.args.grad_accum
            if self.args.use_amp:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            if (batch_idx + 1) % self.args.grad_accum == 0:
                if self.args.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    loss_accum[k] = loss_accum.get(k, 0) + v
            
            if targets:
                self.evaluator.update(predictions, targets, loss_dict)
            
            pbar.set_postfix(loss=f'{total_loss / num_batches:.4f}')
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_dict = {k: v / max(num_batches, 1) for k, v in loss_accum.items()}
        
        eval_results = self.evaluator.compute()
        print(self.evaluator.format_results(eval_results))
        self.evaluator.reset()
        
        return avg_loss, avg_loss_dict
    
    def validate(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                img_t1 = batch['img_t1'].to(self.device)
                img_t2 = batch['img_t2'].to(self.device)
                targets = self._prepare_targets(batch)
                
                predictions, loss, loss_dict = self.model(img_t1, img_t2, targets)
                total_loss += loss.item()
                num_batches += 1
                
                self.evaluator.update(predictions, targets, loss_dict)
        
        avg_loss = total_loss / max(num_batches, 1)
        eval_results = self.evaluator.compute()
        print(f'\n--- Validation Epoch {epoch+1} ---')
        print(f'Val Loss: {avg_loss:.4f}')
        print(self.evaluator.format_results(eval_results))
        print('---')
        
        self.model.train()
        return avg_loss, eval_results
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'args': self.args
        }
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def _load_checkpoint(self, path):
        print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.current_epoch = ckpt['epoch'] + 1
        self.best_loss = ckpt.get('loss', float('inf'))
        if self.scaler and 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])


def main():
    args = parse_args()
    trainer = TrainerV7(args)
    trainer.train()


if __name__ == '__main__':
    main()

