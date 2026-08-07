"""
HICD v6 Training Script
"""

import os
import sys
import argparse
import yaml
import time
from tqdm import tqdm
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_hicd_v6
from datasets import create_dataloader
from evaluation import ICDEvaluatorV6
from datasets.stitcher import PatchStitcher


def parse_args():
    parser = argparse.ArgumentParser(description='HICD v6 Training')
    
    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., xbd, second, 0617final)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root data directory')
    parser.add_argument('--scenes', type=str, default=None,
                        help='Comma-separated list of scenes')
    parser.add_argument('--config_dir', type=str, default='datasets/configs',
                        help='Config directory')
    parser.add_argument('--boxes_dir', type=str, default=None,
                        help='Pre-extracted bounding boxes directory')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Crop patch size')
    parser.add_argument('--patch_dir', type=str, default=None,
                        help='Precomputed patch directory (default: {data_dir}_patches)')
    
    # Model
    parser.add_argument('--backbone', type=str, default='L1',
                        choices=['L0', 'L1', 'L2'],
                        help='LWGANet variant')
    parser.add_argument('--pretrained_weight_path', type=str, default=None,
                        help='Path to pretrained backbone weights')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_queries', type=int, default=100,
                        help='Number of object queries for DETR')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--grad_accum', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='Maximum epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use mixed precision training')
    
    # Training strategy
    parser.add_argument('--freeze_backbone_epochs', type=int, default=10,
                        help='Freeze backbone for first N epochs')
    parser.add_argument('--unfreeze_ratio', type=float, default=0.5,
                        help='Ratio of backbone to unfreeze after freeze stage')
    
    # Loss weights
    parser.add_argument('--lambda_inst', type=float, default=1.0,
                        help='Weight for instance loss')
    parser.add_argument('--lambda_sem', type=float, default=1.0,
                        help='Weight for semantic loss')
    
    
    # Output
    parser.add_argument('--exp_name', type=str, default='v6_experiment',
                        help='Experiment name')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Checkpoint save directory')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Checkpoint save interval')
    
    return parser.parse_args()


class TrainerV6:
    """HICD v6 Trainer"""
    
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
        self.model = build_hicd_v6(model_config).to(self.device)
        
        # Print model summary
        self._print_model_summary()
        
        # Create dataloaders
        scenes = args.scenes.split(',') if args.scenes else None
        self.train_loader = create_dataloader(
            data_dir=args.data_dir,
            config=self.dataset_config,
            split='train',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            scenes=scenes,
            boxes_dir=args.boxes_dir,
            patch_size=args.patch_size,
            patch_dir=getattr(args, 'patch_dir', None)
        )

        self.val_loader = create_dataloader(
            data_dir=args.data_dir,
            config=self.dataset_config,
            split='val',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            scenes=scenes,
            boxes_dir=args.boxes_dir,
            patch_size=args.patch_size,
            patch_dir=getattr(args, 'patch_dir', None)
        )
        
        # Optimizer
        param_groups = self.model.get_param_groups(
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        self.optimizer = optim.AdamW(param_groups)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.max_epochs,
            eta_min=args.learning_rate * 0.01
        )
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if args.use_amp else None
        
        # ICD Evaluator
        self.evaluator = ICDEvaluatorV6(self.dataset_config)
        self.patch_size = args.patch_size
        
        # Save directory
        self.save_dir = os.path.join(args.save_dir, args.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
    
    def _build_model_config(self):
        """Build model configuration"""
        config = {
            'backbone': {
                'variant': self.args.backbone,
                'pretrained': self.args.pretrained_weight_path is not None,
                'pretrained_path': self.args.pretrained_weight_path
            },
            'branch_routing': self.dataset_config.get('branch_routing', {}),
            'clip': {
                'enabled': False,
                'model_path': None
            },
            'num_targets': self.dataset_config.get('num_targets', 1),
            'num_states': self.dataset_config.get('num_states', 4),
            'hidden_dim': self.args.hidden_dim,
            'num_queries': self.args.num_queries,
            'lambda_inst': self.args.lambda_inst,
            'lambda_sem': self.args.lambda_sem,
            'state_names': self.dataset_config.get('state_names', [])
        }
        return config
    
    def _print_model_summary(self):
        """Print model summary"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n{'='*60}")
        print(f"Model Summary")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print(f"{'='*60}\n")
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.args.max_epochs} epochs...")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Gradient accumulation: {self.args.grad_accum}")
        print(f"Effective batch size: {self.args.batch_size * self.args.grad_accum}")
        print(f"Save directory: {self.save_dir}")
        
        for epoch in range(self.args.max_epochs):
            self.current_epoch = epoch
            
            # Training strategy
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
            
            # Train one epoch
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch, train_loss)
            
            # Save best
            if val_metrics < self.best_loss:
                self.best_loss = val_metrics
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            print(f'Epoch {epoch+1}/{self.args.max_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_metrics:.4f} - Best: {self.best_loss:.4f}')
            
        
        print(f"\nTraining completed!")
        print(f"Best loss: {self.best_loss:.4f}")
    
    def validate(self, epoch):
        """Run validation and compute metrics."""
        self.model.eval()
        self.evaluator.reset()
        total_val_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                pre_imgs = batch['img_t1'].to(self.device)
                post_imgs = batch['img_t2'].to(self.device)
                targets = self._prepare_targets(batch)

                predictions, loss, loss_dict = self.model(pre_imgs, post_imgs, targets)
                total_val_loss += loss.item()
                num_batches += 1

                # Debug: print first batch targets
                if num_batches == 1:
                    has_inst = 'instance_labels' in targets
                    n_boxes = sum(len(t.get('boxes', [])) for t in targets.get('instance_labels', [])) if has_inst else 0
                    print(f'  Val batch 1: has_instance_labels={has_inst}, n_boxes={n_boxes}, loss={loss.item():.4f}, loss_dict={loss_dict}')

                self.evaluator.update(predictions, targets, loss_dict)

        avg_val_loss = total_val_loss / max(num_batches, 1)
        eval_results = self.evaluator.compute()
        print(f'\\n--- Validation Epoch {epoch+1} ---')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(self.evaluator.format_results(eval_results))
        print('---')

        self.model.train()
        return avg_val_loss

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        start_time = time.time()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.args.max_epochs}', leave=True, ncols=120, mininterval=0.5, file=sys.stdout)
        for batch_idx, batch in enumerate(pbar):
            img_t1 = batch['img_t1'].to(self.device)
            img_t2 = batch['img_t2'].to(self.device)
            targets = self._prepare_targets(batch)

            if self.args.use_amp:
                with autocast('cuda'):
                    predictions, loss, loss_dict = self.model(img_t1, img_t2, targets)
            else:
                predictions, loss, loss_dict = self.model(img_t1, img_t2, targets)

            if self.args.use_amp:
                scaled_loss = loss / self.args.grad_accum
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss = loss / self.args.grad_accum
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

            if targets:
                self.evaluator.update(predictions, targets, loss_dict)

            pbar.set_postfix(loss=f'{total_loss / num_batches:.4f}')
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute and display ICD metrics
        eval_results = self.evaluator.compute()
        print(self.evaluator.format_results(eval_results))
        self.evaluator.reset()
        
        return avg_loss
    
    def _log_to_csv(self, epoch, train_loss, r):
        """Append one row to CSV log."""
        loss_avg = r.get('loss_avg', {})
        row = [
            epoch + 1,
            f"{train_loss:.4f}",
            f"{r.get('instance_mAP@0.5', 0):.4f}",
            f"{r.get('instance_mAP@0.75', 0):.4f}",
            f"{r.get('instance_mAP@[0.5:0.95]', 0):.4f}",
            f"{loss_avg.get('inst_loss_class', 0):.4f}",
            f"{loss_avg.get('inst_loss_bbox', 0):.4f}",
            f"{loss_avg.get('inst_loss_giou', 0):.4f}",
            f"{loss_avg.get('inst_total', 0):.4f}",
            f"{loss_avg.get('sem_target', 0):.4f}",
            f"{loss_avg.get('sem_dice', 0):.4f}",
            f"{loss_avg.get('sem_total', 0):.4f}",
            f"{r.get('target_miou', 0):.4f}",
            f"{r.get('target_accuracy', 0):.4f}",
        ]
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def _prepare_targets(self, batch):
        """Prepare targets for model"""
        targets = {}
        
        # For instance branch
        if 'instance_boxes' in batch:
            instance_labels = []
            instance_classes = batch.get('instance_classes')
            instance_states = batch.get('instance_states')
            for i, boxes in enumerate(batch['instance_boxes']):
                if boxes is not None and len(boxes) > 0:
                    if instance_classes is not None and instance_classes[i] is not None:
                        labels = instance_classes[i].to(self.device)
                    else:
                        labels = torch.zeros(len(boxes), dtype=torch.long, device=self.device)
                    inst_dict = {
                        'labels': labels,
                        'boxes': boxes.to(self.device)
                    }
                    if instance_states is not None and instance_states[i] is not None:
                        inst_dict['states'] = instance_states[i].to(self.device)
                    instance_labels.append(inst_dict)
                else:
                    instance_labels.append({
                        'labels': torch.zeros(0, dtype=torch.long, device=self.device),
                        'boxes': torch.zeros(0, 4, device=self.device)
                    })
            targets['instance_labels'] = instance_labels
        
        # For semantic branch
        semantic_classes = self.dataset_config.get('branch_routing', {}).get('semantic', [])
        if 'label' in batch and len(semantic_classes) > 0:
            label = batch['label'].to(self.device)
            if label.dim() == 3:
                label = label.unsqueeze(1).float()
                label = torch.nn.functional.interpolate(
                    label, scale_factor=0.25, mode='nearest'
                ).squeeze(1).long()
            targets['target_gt'] = label
            # state_gt not set: no pixel-level state annotation in SECOND
        
        return targets
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save checkpoint"""
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
        
        # Save regular checkpoint
        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  Saved best model with loss {loss:.4f}")


def main():
    args = parse_args()
    
    # Fix config_dir to be relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    if not os.path.isabs(args.config_dir):
        args.config_dir = os.path.join(project_dir, args.config_dir)
    
    # Print arguments
    print("\n" + "="*60)
    print("Arguments:")
    print("="*60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("="*60 + "\n")
    
    # Create trainer and train
    trainer = TrainerV6(args)
    trainer.train()


if __name__ == '__main__':
    main()
















