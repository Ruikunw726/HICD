"""
Semantic Branch Head
FPN + dual heads for target region and state classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Feature Pyramid Network"""
    
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            lateral = nn.Conv2d(in_channels, out_channels, 1)
            output = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.lateral_convs.append(lateral)
            self.output_convs.append(output)
    
    def forward(self, features):
        # Lateral connections
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], scale_factor=2, mode='bilinear', align_corners=True
            )
        
        # Output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        
        return outputs


class SemanticHead(nn.Module):
    """
    Semantic Branch Head with target and state predictions
    
    Args:
        num_targets: Number of target classes
        num_states: Number of state classes
        in_channels: Input channels from FPN
    """
    
    def __init__(self, num_targets, num_states, in_channels=128):
        super().__init__()
        
        self.num_targets = num_targets
        self.num_states = num_states
        
        # FPN
        self.fpn = FPN(
            in_channels_list=[in_channels] * 4,
            out_channels=in_channels
        )
        
        # Target region head
        self.target_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_targets, 1)
        )
        
        # State classification head
        self.state_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_states, 1)
        )
        
        # State embedding projection
        self.state_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )
    
    def forward(self, features, state_embeddings=None):
        """
        Forward pass
        
        Args:
            features: List of multi-scale features [C2, C3, C4, C5]
            state_embeddings: State embeddings [num_states, C] (from LearnableStateEmbeddings)
            
        Returns:
            target_logits: Target region logits [B, num_targets, H, W]
            state_logits: State classification logits [B, num_states, H, W]
        """
        # FPN fusion
        fpn_features = self.fpn(features)
        
        # Use P2 (highest resolution) for prediction
        p2 = fpn_features[0]  # [B, C, H/4, W/4]
        
        # Target prediction
        target_logits = self.target_head(p2)
        
        # State prediction
        state_logits = self.state_head(p2)
        
        # State embedding enhancement
        if state_embeddings is not None:
            # Project state embeddings
            state_proj = self.state_proj(state_embeddings)  # [num_states, C]
            
            # Compute cosine similarity
            B, C, H, W = p2.shape
            p2_flat = p2.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            p2_norm = F.normalize(p2_flat, dim=-1)
            state_norm = F.normalize(state_proj, dim=-1)
            
            similarity = torch.mm(p2_norm, state_norm.t())  # [B*H*W, num_states]
            similarity = similarity.reshape(B, H, W, self.num_states).permute(0, 3, 1, 2)
            
            # Add similarity to state logits
            state_logits = state_logits + similarity
        
        return target_logits, state_logits
