"""
HICD v6: Hierarchical Instance Change Detection v6
Backbone: LWGANet (Light-Weight Grouped Attention Network)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import LWGANetBackbone
from .modules import (
    NeighborFeatureAggregation,
    MultiScaleTemporalFusion,
    MultiScaleAdapter,
    LightweightCLIP,
    LearnableStateEmbeddings
)
from .heads import DETRInstanceHead, SemanticHead
from .losses import DualBranchLoss


class HICD_v6(nn.Module):
    """
    HICD v6 Model
    
    Dual-branch change detection with:
    - LWGANet backbone
    - Neighbor Feature Aggregation
    - Temporal Fusion Module
    - Task-Specific Adapters
    - Instance branch (DETR)
    - Semantic branch (FPN + dual heads)
    - CLIP for state classification
    
    Args:
        config: Configuration dictionary
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Extract config
        backbone_cfg = config.get('backbone', {})
        branch_cfg = config.get('branch_routing', {})
        clip_cfg = config.get('clip', {})
        
        num_targets = config.get('num_targets', 1)
        num_states = config.get('num_states', 4)
        self.num_states = num_states
        hidden_dim = config.get('hidden_dim', 128)
        
        # ============ Backbone ============
        self.backbone = LWGANetBackbone(
            variant=backbone_cfg.get('variant', 'L1'),
            pretrained=backbone_cfg.get('pretrained', True),
            pretrained_path=backbone_cfg.get('pretrained_path', None)
        )
        backbone_channels = self.backbone.get_out_channels()
        
        # ============ Neighbor Feature Aggregation ============
        self.nfa_t1 = NeighborFeatureAggregation(
            in_channels=backbone_channels,
            out_channels=hidden_dim
        )
        self.nfa_t2 = NeighborFeatureAggregation(
            in_channels=backbone_channels,
            out_channels=hidden_dim
        )
        
        # ============ Temporal Fusion ============
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
        
        # ============ Semantic Branch ============
        use_clip = clip_cfg.get('enabled', False)
        self.semantic_head = SemanticHead(
            num_targets=num_targets,
            num_states=num_states,
            in_channels=hidden_dim
        )
        
        # ============ CLIP ============
        self.use_clip = use_clip
        if use_clip:
            clip_model_path = clip_cfg.get('model_path', None)
            if clip_model_path:
                self.clip_encoder = LightweightCLIP(
                    clip_model_path=clip_model_path,
                    out_channels=hidden_dim
                )
            else:
                # Use learnable embeddings as fallback
                self.clip_encoder = LearnableStateEmbeddings(
                    num_states=num_states,
                    embed_dim=hidden_dim
                )
                self.use_clip = False  # No CLIP text features
        
        # ============ Loss ============

        # ============ State Embeddings ============
        self.num_states = num_states
        self.state_embeddings = LearnableStateEmbeddings(
            num_states=num_states,
            embed_dim=hidden_dim
        )

        self.loss_fn = DualBranchLoss(
            num_target_classes=num_targets,
            num_state_classes=num_states,
            lambda_inst=config.get('lambda_inst', 1.0),
            lambda_sem=config.get('lambda_sem', 1.0)
        )
        
        # Branch routing
        self.instance_classes = branch_cfg.get('instance', [])
        self.semantic_classes = branch_cfg.get('semantic', [])
        
        # State prompts for CLIP
        self.state_prompts = config.get('state_names', [])
    
    def forward(self, img_t1, img_t2, targets=None):
        """
        Forward pass
        
        Args:
            img_t1: Image at time 1 [B, 3, H, W]
            img_t2: Image at time 2 [B, 3, H, W]
            targets: Ground truth dict (for training)
            
        Returns:
            predictions: dict with predictions
            loss: Loss value (if targets provided)
        """
        B = img_t1.shape[0]
        
        # ============ Extract Features ============
        # Backbone
        features_t1 = self.backbone(img_t1)  # [C2, C3, C4, C5]
        features_t2 = self.backbone(img_t2)  # [C2, C3, C4, C5]
        
        # Neighbor Feature Aggregation
        nfa_t1 = self.nfa_t1(*features_t1)  # (S2, S3, S4, S5)
        nfa_t2 = self.nfa_t2(*features_t2)  # (S2, S3, S4, S5)
        
        # Temporal Fusion
        fused_features = self.tfm(nfa_t1, nfa_t2)  # [C2, C3, C4, C5]
        
        # ============ Task-Specific Adapters ============
        instance_features = self.instance_adapter(fused_features)
        
        # ============ Instance Branch ============
        instance_feat = instance_features[0]  # [B, C, H/4, W/4]
        instance_logits, instance_boxes, instance_state_logits = self.instance_head(instance_feat)
        
        predictions = {
            'instance_logits': instance_logits,
            'instance_boxes': instance_boxes,
        }
        if instance_state_logits is not None:
            predictions['instance_state_logits'] = instance_state_logits
        
        # ============ Semantic Branch (skip if no semantic classes) ============
        if self.semantic_classes:
            semantic_features = self.semantic_adapter(fused_features)
            state_ids = torch.arange(self.num_states, device=img_t1.device)
            clip_text_features = self.state_embeddings(state_ids)
            target_logits, state_logits = self.semantic_head(
                semantic_features,
                state_embeddings=clip_text_features
            )
            predictions['target_logits'] = target_logits
            predictions['state_logits'] = state_logits
        
        # ============ Loss ============
        loss = None
        if targets is not None:
            loss, loss_dict = self.loss_fn(predictions, targets)
            return predictions, loss, loss_dict
        
        return predictions
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        self.backbone.freeze()
    
    def unfreeze_backbone(self, ratio=1.0):
        """Unfreeze backbone parameters"""
        self.backbone.unfreeze(ratio)
    
    def get_param_groups(self, lr=1e-4, weight_decay=1e-4):
        """
        Get parameter groups for optimizer
        
        Returns:
            List of parameter groups with different learning rates
        """
        param_groups = []
        
        # Backbone (lower learning rate)
        backbone_params = list(self.backbone.parameters())
        param_groups.append({
            'params': backbone_params,
            'lr': lr * 0.1,
            'weight_decay': weight_decay,
            'name': 'backbone'
        })
        
        # NFA + TFM
        ntfm_params = list(self.nfa_t1.parameters()) + \
                      list(self.nfa_t2.parameters()) + \
                      list(self.tfm.parameters())
        param_groups.append({
            'params': ntfm_params,
            'lr': lr,
            'weight_decay': weight_decay,
            'name': 'nfa_tfm'
        })
        
        # Adapters
        adapter_params = list(self.instance_adapter.parameters()) + \
                        list(self.semantic_adapter.parameters())
        param_groups.append({
            'params': adapter_params,
            'lr': lr,
            'weight_decay': weight_decay,
            'name': 'adapters'
        })
        
        # Instance head
        instance_params = list(self.instance_head.parameters())
        param_groups.append({
            'params': instance_params,
            'lr': lr,
            'weight_decay': weight_decay,
            'name': 'instance_head'
        })
        
        # Semantic head
        semantic_params = list(self.semantic_head.parameters())
        param_groups.append({
            'params': semantic_params,
            'lr': lr,
            'weight_decay': weight_decay,
            'name': 'semantic_head'
        })
        
        return param_groups


def build_hicd_v6(config):
    """
    Build HICD v6 model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HICD v6 model
    """
    return HICD_v6(config)


if __name__ == '__main__':
    # Test
    config = {
        'backbone': {
            'variant': 'L1',
            'pretrained': False,
            'pretrained_path': None
        },
        'branch_routing': {
            'instance': ['Building'],
            'semantic': ['Non-vegetation', 'Tree', 'Low vegetation', 'Water']
        },
        'clip': {
            'enabled': False,
            'model_path': None
        },
        'num_targets': 2,
        'num_states': 5,
        'hidden_dim': 128,
        'num_queries': 50,
        'state_names': ['no-damage', 'minor-damage', 'major-damage', 'destroyed', 'un-classified']
    }
    
    model = HICD_v6(config)
    
    img_t1 = torch.randn(2, 3, 256, 256)
    img_t2 = torch.randn(2, 3, 256, 256)
    
    predictions = model(img_t1, img_t2)
    print("Predictions:")
    for k, v in predictions.items():
        print(f"  {k}: {v.shape}")






