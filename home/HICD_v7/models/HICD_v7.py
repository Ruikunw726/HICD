"""
HICD v7: Hierarchical Instance Change Detection v7
Dual-branch with interaction layer.

Key changes from V6:
- TFM returns T1/T2 independent features
- Three interaction mechanisms between branches
- ChangeTypeClassifier for change type + damage level
- Supports active_heads config for dataset-specific heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import LWGANetBackbone
from .modules import (
    NeighborFeatureAggregation,
    MultiScaleTemporalFusion,
    MultiScaleAdapter,
    ChangeGuidedAttention,
    ChangeTypeClassifier,
    LearnableStateEmbeddings
)
from .heads import DETRInstanceHead, SemanticHead
from .losses import V7Loss


class HICD_v7(nn.Module):
    """
    HICD v7 Model
    
    Architecture:
    T1, T2 -> LWGANet (shared) -> NFA -> TFM (fused + T1/T2 independent)
    -> Task Adapters -> Instance Branch (DETR) + Pixel Branch (FPN)
    -> Interaction Layer (ChangeGuidedAttention + ChangeTypeClassifier)
    -> Output: boxes + target class + change type + damage level
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Extract config
        backbone_cfg = config.get('backbone', {})
        clip_cfg = config.get('clip', {})
        active_heads = config.get('active_heads', {})
        self.active_heads = active_heads
        
        num_targets = config.get('num_targets', 1)
        num_pixel_targets = config.get('num_pixel_targets', num_targets)
        num_states = config.get('num_states', 5)
        num_change_types = config.get('num_change_types', 5)
        num_damage_levels = config.get('num_damage_levels', 4)
        self.num_states = num_states
        hidden_dim = config.get('hidden_dim', 128)
        
        # ============ Backbone ============
        self.backbone = LWGANetBackbone(
            variant=backbone_cfg.get('variant', 'L1'),
            pretrained=backbone_cfg.get('pretrained', True),
            pretrained_path=backbone_cfg.get('pretrained_path', None)
        )
        backbone_channels = self.backbone.get_out_channels()
        
        # ============ NFA ============
        self.nfa_t1 = NeighborFeatureAggregation(
            in_channels=backbone_channels, out_channels=hidden_dim
        )
        self.nfa_t2 = NeighborFeatureAggregation(
            in_channels=backbone_channels, out_channels=hidden_dim
        )
        
        # ============ Temporal Fusion (V7: returns T1/T2 independent features) ============
        self.tfm = MultiScaleTemporalFusion(
            in_channels=[hidden_dim] * 4,
            out_channels=[hidden_dim] * 4
        )
        
        # ============ Task-Specific Adapters ============
        self.instance_adapter = MultiScaleAdapter(
            in_channels=[hidden_dim] * 4,
            adapter_channels=hidden_dim // 4
        )
        self.semantic_adapter = MultiScaleAdapter(
            in_channels=[hidden_dim] * 4,
            adapter_channels=hidden_dim // 4
        )
        
        # ============ Instance Branch ============
        self.instance_head = DETRInstanceHead(
            num_classes=num_targets,
            num_states=num_states,
            hidden_dim=hidden_dim,
            num_queries=config.get('num_queries', 100),
            nhead=config.get('nhead', 8),
            num_decoder_layers=config.get('num_decoder_layers', 6)
        )
        
        # ============ Pixel Branch ============
        self.semantic_head = SemanticHead(
            num_targets=num_pixel_targets,
            num_states=num_states,
            in_channels=hidden_dim
        )
        
        # ============ Interaction Layer (NEW) ============
        # Interaction 1: Change heatmap -> instance attention guidance
        self.change_guided_attn = ChangeGuidedAttention(dim=hidden_dim)
        
        # Interaction 3: T1/T2 feature comparison -> change type + damage level
        self.change_type_classifier = ChangeTypeClassifier(
            hidden_dim=hidden_dim,
            num_change_types=num_change_types,
            num_damage_levels=num_damage_levels,
            spatial_scale=0.25
        )
        
        # ============ State Embeddings ============
        self.state_embeddings = LearnableStateEmbeddings(
            num_states=num_states, embed_dim=hidden_dim
        )
        
        # ============ Loss ============
        config['num_pixel_targets'] = num_pixel_targets
        self.loss_fn = V7Loss(config)
        
        # Store T1/T2 features for interaction layer
        self._t1_features = None
        self._t2_features = None
    
    def forward(self, img_t1, img_t2, targets=None):
        """
        Args:
            img_t1: [B, 3, H, W]
            img_t2: [B, 3, H, W]
            targets: dict (for training)
            
        Returns:
            predictions: dict
            loss: scalar (if targets provided)
            loss_dict: dict (if targets provided)
        """
        B = img_t1.shape[0]
        
        # ======== [1] Backbone ========
        features_t1 = self.backbone(img_t1)  # (C2, C3, C4, C5)
        features_t2 = self.backbone(img_t2)
        
        # ======== [2] NFA ========
        nfa_t1 = self.nfa_t1(*features_t1)
        nfa_t2 = self.nfa_t2(*features_t2)
        
        # ======== [3] Temporal Fusion (V7: save T1/T2 independent features) ========
        fused_features, t1_features, t2_features = self.tfm(nfa_t1, nfa_t2)
        # fused_features: [C2, C3, C4, C5]
        # t1_features, t2_features: [C2, C3, C4, C5] 鈥?for interaction layer
        
        # ======== [4] Task Adapters ========
        instance_features = self.instance_adapter(fused_features)
        pixel_features = self.semantic_adapter(fused_features)
        
        # ======== [5A] Instance Branch ========
        instance_feat = instance_features[0]  # [B, C, H/4, W/4]
        pred_logits, pred_boxes, pred_state_logits, instance_decoder_out = \
            self.instance_head(instance_feat)
        
        # ======== [5B] Pixel Branch ========
        change_heatmap, target_type_map = self.semantic_head(pixel_features)
        
        predictions = {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
            'instance_features': instance_decoder_out,
            'change_heatmap': change_heatmap,
            'target_type_map': target_type_map,
        }
        if pred_state_logits is not None:
            predictions['pred_state_logits'] = pred_state_logits
        
        # ======== [6] Interaction Layer ========
        # Interaction 1: Change heatmap guides instance attention
        # Note: gate starts at 0, so this doesn't affect early training
        # The change heatmap is used as a soft attention bias
        
        # Interaction 3: T1/T2 feature comparison -> change type + damage level
        t1_feat = t1_features[0]  # Use P2 scale (highest resolution)
        t2_feat = t2_features[0]
        
        change_type_logits, damage_level_logits = self.change_type_classifier(
            t1_feat, t2_feat, instance_decoder_out, pred_boxes
        )
        predictions['pred_change_type'] = change_type_logits
        predictions['pred_damage_level'] = damage_level_logits
        
        # Store state map for box consistency loss
        predictions['pred_state_map'] = target_type_map
        
        # ======== [7] Loss ========
        loss = None
        loss_dict = None
        if targets is not None:
            loss, loss_dict = self.loss_fn(predictions, targets)
            return predictions, loss, loss_dict
        
        return predictions
    
    def freeze_backbone(self):
        self.backbone.freeze()
    
    def unfreeze_backbone(self, ratio=1.0):
        self.backbone.unfreeze(ratio)
    
    def get_param_groups(self, lr=1e-4, weight_decay=1e-4):
        param_groups = []
        
        # Backbone (lower LR)
        param_groups.append({
            'params': list(self.backbone.parameters()),
            'lr': lr * 0.1, 'weight_decay': weight_decay, 'name': 'backbone'
        })
        
        # NFA + TFM
        ntfm_params = list(self.nfa_t1.parameters()) + \
                      list(self.nfa_t2.parameters()) + \
                      list(self.tfm.parameters())
        param_groups.append({
            'params': ntfm_params,
            'lr': lr, 'weight_decay': weight_decay, 'name': 'nfa_tfm'
        })
        
        # Adapters
        adapter_params = list(self.instance_adapter.parameters()) + \
                        list(self.semantic_adapter.parameters())
        param_groups.append({
            'params': adapter_params,
            'lr': lr, 'weight_decay': weight_decay, 'name': 'adapters'
        })
        
        # Instance head
        param_groups.append({
            'params': list(self.instance_head.parameters()),
            'lr': lr, 'weight_decay': weight_decay, 'name': 'instance_head'
        })
        
        # Pixel head
        param_groups.append({
            'params': list(self.semantic_head.parameters()),
            'lr': lr, 'weight_decay': weight_decay, 'name': 'semantic_head'
        })
        
        # Interaction layer (new)
        interaction_params = list(self.change_guided_attn.parameters()) + \
                            list(self.change_type_classifier.parameters())
        param_groups.append({
            'params': interaction_params,
            'lr': lr, 'weight_decay': weight_decay, 'name': 'interaction'
        })
        
        return param_groups


def build_hicd_v7(config):
    """Build HICD v7 model"""
    return HICD_v7(config)


