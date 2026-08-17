"""
Dual Branch Loss
Combined loss for instance and semantic branches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_zero_loss(predictions, key, device=None):
    """Return zero loss with grad, protecting against NaN in predictions."""
    if key in predictions:
        val = predictions[key]
        if torch.isnan(val).any():
            val = torch.nan_to_num(val, nan=0.0)
        return val.sum() * 0.0
    if device is None:
        for v in predictions.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
    return torch.tensor(0.0, device=device, requires_grad=True)


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Compute Dice Loss
        
        Args:
            pred: [B, C, H, W] logits
            target: [B, H, W] or [B, 1, H, W] ground truth
            
        Returns:
            Dice loss
        """
        pred = F.softmax(pred, dim=1)
        
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # One-hot encode target
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target.long().squeeze(1), num_classes)  # [B, H, W, C]
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Flatten
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
        target_flat = target_onehot.view(target_onehot.shape[0], target_onehot.shape[1], -1)
        
        # Compute dice
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice.mean()


class DualBranchLoss(nn.Module):
    """
    Dual Branch Loss for HICD v6
    
    Args:
        num_target_classes: Number of target classes for semantic branch
        num_state_classes: Number of state classes for semantic branch
        lambda_inst: Weight for instance loss
        lambda_sem: Weight for semantic loss
        lambda_target: Weight for target loss within semantic
        lambda_state: Weight for state loss within semantic
        lambda_dice: Weight for dice loss within semantic
    """
    
    def __init__(self, num_target_classes, num_state_classes,
                 lambda_inst=1.0, lambda_sem=1.0,
                 lambda_target=1.0, lambda_state=1.0, lambda_dice=1.0):
        super().__init__()
        
        from .detr_loss import DETRLoss
        
        # Instance loss (DETR)
        self.instance_loss = DETRLoss(num_classes=num_target_classes, num_states=num_state_classes)
        
        # Semantic losses
        self.target_loss = nn.CrossEntropyLoss()
        self.state_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        
        # Weights
        self.lambda_inst = lambda_inst
        self.lambda_sem = lambda_sem
        self.lambda_target = lambda_target
        self.lambda_state = lambda_state
        self.lambda_dice = lambda_dice
    
    def forward(self, predictions, targets):
        """
        Compute dual branch loss
        
        Args:
            predictions: dict with:
                - instance_logits: [B, num_queries, num_classes + 1]
                - instance_boxes: [B, num_queries, 4]
                - target_logits: [B, num_targets, H, W]
                - state_logits: [B, num_states, H, W]
            targets: dict with:
                - instance_labels: list of dicts with 'labels' and 'boxes'
                - target_gt: [B, H, W]
                - state_gt: [B, H, W]
                
        Returns:
            total_loss: Total weighted loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}
        
        # Instance loss
        if 'instance_logits' in predictions and 'instance_labels' in targets:
            pred_state = predictions.get('instance_state_logits')
            loss_inst, inst_dict = self.instance_loss(
                predictions['instance_logits'],
                predictions['instance_boxes'],
                targets['instance_labels'],
                pred_state=pred_state
            )
            loss_dict.update({f'inst_{k}': v for k, v in inst_dict.items()})
        else:
            # No instance labels - use zero loss with grad
            loss_inst = _safe_zero_loss(predictions, 'instance_logits')
        
        # Target loss
        if 'target_logits' in predictions and 'target_gt' in targets:
            loss_target = self.target_loss(predictions['target_logits'], targets['target_gt'])
            loss_dict['sem_target'] = loss_target.item()
        else:
            loss_target = _safe_zero_loss(predictions, 'target_logits')
        
        # State loss
        if 'state_logits' in predictions and 'state_gt' in targets:
            loss_state = self.state_loss(predictions['state_logits'], targets['state_gt'])
            loss_dict['sem_state'] = loss_state.item()
        else:
            loss_state = _safe_zero_loss(predictions, 'state_logits')
        
        # Dice loss
        if 'target_logits' in predictions and 'target_gt' in targets:
            loss_dice = self.dice_loss(predictions['target_logits'], targets['target_gt'])
            loss_dict['sem_dice'] = loss_dice.item()
        else:
            loss_dice = _safe_zero_loss(predictions, 'target_logits')
        
        # Semantic loss
        loss_sem = (self.lambda_target * loss_target + 
                    self.lambda_state * loss_state + 
                    self.lambda_dice * loss_dice)
        
        # Total loss
        total_loss = self.lambda_inst * loss_inst + self.lambda_sem * loss_sem
        
        total_loss = torch.nan_to_num(total_loss, nan=0.0)
        loss_dict['total'] = total_loss.item()
        loss_dict['inst_total'] = loss_inst.item() if isinstance(loss_inst, torch.Tensor) else loss_inst
        loss_dict['sem_total'] = loss_sem.item()
        
        return total_loss, loss_dict


if __name__ == '__main__':
    # Test
    loss_fn = DualBranchLoss(
        num_target_classes=4,
        num_state_classes=5,
        lambda_inst=1.0,
        lambda_sem=1.0
    )
    
    predictions = {
        'instance_logits': torch.randn(2, 50, 3),
        'instance_boxes': torch.rand(2, 50, 4),
        'target_logits': torch.randn(2, 4, 64, 64),
        'state_logits': torch.randn(2, 5, 64, 64)
    }
    
    targets = {
        'instance_labels': [
            {'labels': torch.tensor([0, 1]), 'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.3, 0.3, 0.2, 0.2]])},
            {'labels': torch.tensor([0]), 'boxes': torch.tensor([[0.4, 0.4, 0.15, 0.15]])}
        ],
        'target_gt': torch.randint(0, 4, (2, 64, 64)),
        'state_gt': torch.randint(0, 5, (2, 64, 64))
    }
    
    total_loss, loss_dict = loss_fn(predictions, targets)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")

