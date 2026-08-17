"""
V7 Interaction Layer
Three interaction mechanisms between instance and pixel branches:
1. ChangeGuidedAttention: change heatmap guides instance branch attention
2. BoxConsistencyLoss: detection boxes constrain pixel branch consistency
3. ChangeTypeClassifier: T1/T2 feature comparison �� change type + damage level
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class ChangeGuidedAttention(nn.Module):
    """
    Interaction 1: Change heatmap guides instance branch attention.
    
    The pixel branch's change heatmap is upsampled to instance feature resolution
    and added (with a learnable gate) to the DETR decoder's cross-attention map.
    
    Gate is initialized to 0 so training early doesn't interfere with DETR attention.
    """
    
    def __init__(self, dim=128):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
    
    def forward(self, attn_map, change_heatmap):
        """
        Args:
            attn_map: (B, num_queries, H, W) �� DETR cross-attention map (reshaped)
            change_heatmap: (B, 1, H/4, W/4) �� pixel branch output
            
        Returns:
            attn_map_enhanced: (B, num_queries, H, W)
        """
        # Upsample change heatmap to match attention map resolution
        change_weight = self.upsample(change_heatmap)  # (B, 1, H, W)
        
        # Gate-controlled fusion
        attn_map_enhanced = attn_map + self.gate * change_weight
        return attn_map_enhanced


class BoxConsistencyLoss(nn.Module):
    """
    Interaction 2: Detection boxes constrain pixel branch consistency.
    
    Within each detected box, pixels should belong to the same change state.
    This encourages the pixel branch to produce consistent predictions within
    each instance boundary.
    
    Uses RoI Align to extract per-box pixel predictions, then penalizes variance.
    """
    
    def __init__(self, spatial_scale=0.25):
        super().__init__()
        self.spatial_scale = spatial_scale
    
    def forward(self, pred_state_map, gt_boxes, gt_labels):
        """
        Args:
            pred_state_map: (B, N_states, H, W) �� pixel branch state predictions
            gt_boxes: list of (M_i, 4) tensors in [cx, cy, w, h] normalized format
            gt_labels: list of (M_i,) tensors
            
        Returns:
            Scalar consistency loss
        """
        B, C, H, W = pred_state_map.shape
        if B == 0 or pred_state_map.numel() == 0:
            return torch.tensor(0.0, device=pred_state_map.device, requires_grad=True)
        
        total_loss = torch.tensor(0.0, device=pred_state_map.device, requires_grad=True)
        count = 0
        
        for b in range(B):
            boxes = gt_boxes[b]  # (M, 4) in [cx, cy, w, h] normalized
            if boxes is None or len(boxes) == 0:
                continue
            
            # Convert [cx, cy, w, h] normalized �� [x1, y1, x2, y2] pixel coords
            boxes_abs = boxes.clone()
            boxes_abs[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
            boxes_abs[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
            boxes_abs[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
            boxes_abs[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H
            
            # Add batch index
            batch_idx = torch.full((len(boxes), 1), b, dtype=boxes.dtype, device=boxes.device)
            rois = torch.cat([batch_idx, boxes_abs], dim=1)  # (M, 5)
            
            # RoI Align: extract per-box state predictions
            # pred_state_map: (B, C, H, W), rois: (M, 5)
            pooled = roi_align(
                pred_state_map, rois,
                output_size=(7, 7),
                spatial_scale=self.spatial_scale,
                aligned=True
            )  # (M, C, 7, 7)
            
            # Compute per-box variance: lower variance = more consistent
            # Use softmax to get probabilities
            pooled_prob = F.softmax(pooled, dim=1)  # (M, C, 7, 7)
            
            # Variance across spatial locations for each class
            var = pooled_prob.var(dim=(2, 3))  # (M, C)
            total_loss = total_loss + var.mean()
            count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        # Ensure gradient flows
        if not total_loss.requires_grad:
            total_loss = total_loss.clone().requires_grad_(True)
        
        return total_loss


class ChangeTypeClassifier(nn.Module):
    """
    Interaction 3: T1/T2 feature comparison �� change type + damage level.
    
    For each detected instance, extract T1 and T2 features via RoI Align,
    compute difference features, and classify change type and damage level.
    
    Change types: unchanged, new_construction, expansion, destruction, damage
    Damage levels: no_damage, minor_damage, major_damage, destroyed
    """
    
    def __init__(self, hidden_dim=128, num_change_types=5, num_damage_levels=4,
                 spatial_scale=0.25):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_change_types = num_change_types
        self.num_damage_levels = num_damage_levels
        self.spatial_scale = spatial_scale
        
        # Feature extraction via RoI Align
        self.roi_size = 7
        
        # Change type classifier: takes [T1_feat, T2_feat, diff_feat, instance_feat]
        self.change_type_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_change_types)
        )
        
        # Damage level classifier: same input structure
        self.damage_level_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_damage_levels)
        )
    
    def forward(self, t1_features, t2_features, instance_features, pred_boxes):
        """
        Args:
            t1_features: (B, C, H, W) �� T1 independent features from TFM
            t2_features: (B, C, H, W) �� T2 independent features from TFM
            instance_features: (B, num_queries, D) �� DETR decoder output
            pred_boxes: (B, num_queries, 4) �� predicted boxes in [cx, cy, w, h] normalized
            
        Returns:
            change_type_logits: (B, num_queries, num_change_types)
            damage_level_logits: (B, num_queries, num_damage_levels)
        """
        B, num_queries, D = instance_features.shape
        _, C, H, W = t1_features.shape
        
        # Convert boxes to RoI format
        # pred_boxes: [cx, cy, w, h] normalized �� [x1, y1, x2, y2] pixel coords
        boxes_abs = pred_boxes.clone()
        boxes_abs[:, :, 0] = (pred_boxes[:, :, 0] - pred_boxes[:, :, 2] / 2) * W
        boxes_abs[:, :, 1] = (pred_boxes[:, :, 1] - pred_boxes[:, :, 3] / 2) * H
        boxes_abs[:, :, 2] = (pred_boxes[:, :, 0] + pred_boxes[:, :, 2] / 2) * W
        boxes_abs[:, :, 3] = (pred_boxes[:, :, 1] + pred_boxes[:, :, 3] / 2) * H
        
        # Create RoIs with batch index
        batch_idx = torch.arange(B, device=pred_boxes.device).view(B, 1, 1).expand(-1, num_queries, 1)
        rois = torch.cat([batch_idx.float(), boxes_abs], dim=2)  # (B, num_queries, 5)
        rois = rois.reshape(-1, 5)  # (B*num_queries, 5)
        
        # RoI Align on T1 and T2 features
        t1_pooled = roi_align(t1_features, rois, output_size=(self.roi_size, self.roi_size),
                              spatial_scale=self.spatial_scale, aligned=True)
        t2_pooled = roi_align(t2_features, rois, output_size=(self.roi_size, self.roi_size),
                              spatial_scale=self.spatial_scale, aligned=True)
        
        # Global average pooling
        t1_vec = t1_pooled.mean(dim=(2, 3))  # (B*num_queries, C)
        t2_vec = t2_pooled.mean(dim=(2, 3))  # (B*num_queries, C)
        
        # Difference features
        diff_vec = torch.abs(t1_vec - t2_vec)  # (B*num_queries, C)
        
        # Instance features (flatten)
        inst_vec = instance_features.reshape(-1, D)  # (B*num_queries, D)
        
        # Concatenate: [T1, T2, diff, instance]
        combined = torch.cat([t1_vec, t2_vec, diff_vec, inst_vec], dim=1)  # (B*Nq, 4*C)
        
        # Classify
        change_type_logits = self.change_type_head(combined)  # (B*Nq, num_change_types)
        damage_level_logits = self.damage_level_head(combined)  # (B*Nq, num_damage_levels)
        
        # Reshape back
        change_type_logits = change_type_logits.reshape(B, num_queries, self.num_change_types)
        damage_level_logits = damage_level_logits.reshape(B, num_queries, self.num_damage_levels)
        
        return change_type_logits, damage_level_logits
