"""
Temporal Fusion Module (TFM)
Fuse features from two time phases (T1 and T2)
"""

import torch
import torch.nn as nn


class TemporalFusionModule(nn.Module):
    """
    Temporal Fusion Module for change detection
    
    Args:
        in_channels: Input channels for each scale
        out_channels: Output channels for each scale
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # T1 branch
        self.conv_t1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # T2 branch
        self.conv_t2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, feat_t1, feat_t2):
        """
        Forward pass
        
        Args:
            feat_t1: Features from time 1 [B, C, H, W]
            feat_t2: Features from time 2 [B, C, H, W]
            
        Returns:
            Fused features [B, out_channels, H, W]
        """
        feat_t1 = self.conv_t1(feat_t1)
        feat_t2 = self.conv_t2(feat_t2)
        return self.conv_fuse(torch.cat([feat_t1, feat_t2], dim=1))


class MultiScaleTemporalFusion(nn.Module):
    """
    Multi-scale Temporal Fusion Module
    
    Args:
        in_channels: List of input channels for each scale
        out_channels: List of output channels for each scale
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.tfms = nn.ModuleList([
            TemporalFusionModule(in_ch, out_ch)
            for in_ch, out_ch in zip(in_channels, out_channels)
        ])
    
    def forward(self, features_t1, features_t2):
        """
        Forward pass
        
        Args:
            features_t1: List of features from time 1 [C2, C3, C4, C5]
            features_t2: List of features from time 2 [C2, C3, C4, C5]
            
        Returns:
            List of fused features [C2, C3, C4, C5]
        """
        fused_features = []
        for tfm, feat_t1, feat_t2 in zip(self.tfms, features_t1, features_t2):
            fused_features.append(tfm(feat_t1, feat_t2))
        
        return fused_features


if __name__ == '__main__':
    # Test
    tfm = TemporalFusionModule(in_channels=128, out_channels=128)
    
    feat_t1 = torch.randn(2, 128, 32, 32)
    feat_t2 = torch.randn(2, 128, 32, 32)
    
    fused = tfm(feat_t1, feat_t2)
    print(f"Fused: {fused.shape}")
    
    # Multi-scale test
    mtfm = MultiScaleTemporalFusion(
        in_channels=[128, 128, 128, 128],
        out_channels=[128, 128, 128, 128]
    )
    
    features_t1 = [
        torch.randn(2, 128, 64, 64),
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 128, 16, 16),
        torch.randn(2, 128, 8, 8)
    ]
    features_t2 = [f.clone() for f in features_t1]
    
    fused_features = mtfm(features_t1, features_t2)
    for i, feat in enumerate(fused_features):
        print(f"Scale {i+2}: {feat.shape}")
