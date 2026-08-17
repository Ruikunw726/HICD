"""
V7 Dual Branch Loss
Combined loss for instance branch, pixel branch, change type, damage level,
and box consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .detr_loss import DETRLoss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target.long().squeeze(1), num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
        target_flat = target_onehot.view(target_onehot.shape[0], target_onehot.shape[1], -1)
        
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class V7Loss(nn.Module):
    """
    V7 Loss Function
    
    Components:
    1. Instance loss (DETR): class + bbox + giou + state
    2. Pixel loss: target CE + state CE + dice
    3. Change type loss (new)
    4. Damage level loss (new)
    5. Box consistency loss (new)
    """
    
    def __init__(self, config):
        super().__init__()
        
        num_targets = config.get('num_targets', 1)
        num_states = config.get('num_states', 5)
        num_change_types = config.get('num_change_types', 5)
        num_damage_levels = config.get('num_damage_levels', 4)
        
        # Instance loss (DETR)
        self.instance_loss = DETRLoss(
            num_classes=num_targets,
            num_states=num_states,
            loss_class=1.0,
            loss_bbox=5.0,
            loss_giou=2.0,
            loss_state=2.0
        )
        
        # Pixel losses
        self.pixel_target_ce = nn.CrossEntropyLoss()
        self.pixel_state_ce = nn.CrossEntropyLoss()
        self.pixel_dice = DiceLoss()
        
        # Change type loss
        self.change_type_ce = nn.CrossEntropyLoss()
        
        # Damage level loss
        self.damage_level_ce = nn.CrossEntropyLoss()
        
        # Weights
        self.w_instance = config.get('w_instance', 1.0)
        self.w_pixel = config.get('w_pixel', 1.0)
        self.w_change_type = config.get('w_change_type', 1.0)
        self.w_damage_level = config.get('w_damage_level', 1.0)
        self.w_consistency = config.get('w_consistency', 0.3)
        
        # Active heads config
        self.active_heads = config.get('active_heads', {})
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict from HICD_v7.forward()
            targets: dict from dataset
            
        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses
        """
        loss_dict = {}
        device = predictions['pred_boxes'].device
        
        # === 1. Instance loss ===
        loss_instance = torch.tensor(0.0, device=device)
        if 'instance_labels' in targets:
            pred_state = predictions.get('pred_state_logits')
            loss_instance, inst_dict = self.instance_loss(
                predictions['pred_logits'],
                predictions['pred_boxes'],
                targets['instance_labels'],
                pred_state=pred_state
            )
            loss_dict.update({f'inst_{k}': v for k, v in inst_dict.items()})
        loss_dict['loss_instance'] = loss_instance.item()
        
        # === 2. Pixel loss ===
        loss_pixel = torch.tensor(0.0, device=device)
        if self.active_heads.get('pixel_target', False) and 'target_gt' in targets:
            loss_target_ce = self.pixel_target_ce(
                predictions.get('change_heatmap', predictions.get('target_logits')),
                targets['target_gt']
            )
            loss_dice = self.pixel_dice(
                predictions.get('target_type_map', predictions.get('target_logits')),
                targets['target_gt']
            )
            loss_pixel = loss_target_ce + loss_dice
            loss_dict['pixel_target_ce'] = loss_target_ce.item()
            loss_dict['pixel_dice'] = loss_dice.item()
        loss_dict['loss_pixel'] = loss_pixel.item()
        
        # === 3. Change type loss ===
        loss_change_type = torch.tensor(0.0, device=device)
        if (self.active_heads.get('change_type', False) and 
            'change_type_gt' in targets and 'pred_change_type' in predictions):
            loss_change_type = self.change_type_ce(
                predictions['pred_change_type'],
                targets['change_type_gt']
            )
        loss_dict['loss_change_type'] = loss_change_type.item()
        
        # === 4. Damage level loss ===
        loss_damage_level = torch.tensor(0.0, device=device)
        if (self.active_heads.get('damage_level', False) and 
            'damage_level_gt' in targets and 'pred_damage_level' in predictions):
            # Only compute for instances with damage
            damage_gt = targets['damage_level_gt']
            valid_mask = damage_gt >= 0  # -1 means no damage label
            if valid_mask.any():
                loss_damage_level = self.damage_level_ce(
                    predictions['pred_damage_level'][valid_mask],
                    damage_gt[valid_mask]
                )
        loss_dict['loss_damage_level'] = loss_damage_level.item()
        
        # === 5. Box consistency loss ===
        loss_consistency = torch.tensor(0.0, device=device)
        if 'pred_state_map' in predictions and 'instance_labels' in targets:
            gt_boxes_list = [t['boxes'] for t in targets['instance_labels']]
            gt_labels_list = [t['labels'] for t in targets['instance_labels']]
            from ..modules.interaction import BoxConsistencyLoss
            consistency_fn = BoxConsistencyLoss()
            loss_consistency = consistency_fn(
                predictions['pred_state_map'],
                gt_boxes_list,
                gt_labels_list
            )
        loss_dict['loss_consistency'] = loss_consistency.item()
        
        # === Total ===
        total_loss = (
            self.w_instance * loss_instance +
            self.w_pixel * loss_pixel +
            self.w_change_type * loss_change_type +
            self.w_damage_level * loss_damage_level +
            self.w_consistency * loss_consistency
        )
        
        total_loss = torch.nan_to_num(total_loss, nan=0.0)
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
