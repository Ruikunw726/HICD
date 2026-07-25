import torch
import torch.nn as nn
import torch.nn.functional as F
from MambaCD.changedetection.models.Mamba_backbone import Backbone_VSSM
from MambaCD.classification.models.vmamba import LayerNorm2d
from MambaCD.changedetection.models.ChangeDecoder import ChangeDecoder
from MambaCD.changedetection.models.CLIPTextEncoder import CLIPTextEncoder
from MambaCD.changedetection.models.CrossAttentionFusion import TextVisualCrossAttention
from MambaCD.changedetection.models.InstanceDetectionHead import InstanceDetectionHead


class STMambaSCD_Instance(nn.Module):
    """
    纯实例级变化检测模型。

    架构:
        双时相图像 -> 共享 Backbone_VSSM -> ChangeDecoder (像素级特征聚合)
        -> CLIP 交叉注意力增强 -> 实例检测头 -> bbox + 损毁类别

    输出:
        pred_boxes:   (B, num_queries, 4)           实例边界框 [cx, cy, w, h]
        pred_logits:  (B, num_queries, num_classes+1) 实例分类
        instance_feats: (B, num_queries, 128)         实例特征
    """
    def __init__(self, output_classes, pretrained,
                 num_queries=100, clip_model="ViT-B-16",
                 clip_weights_path=None, text_prompts=None, **kwargs):
        super().__init__()

        # ========== 编码器 (共享权重 Siamese) ==========
        self.encoder = Backbone_VSSM(
            out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs
        )

        _NORMLAYERS = dict(ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)

        self.channel_first = self.encoder.channel_first
        norm_layer = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        ssm_act_layer = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)
        clean_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']
        }

        # ========== 变化解码器 (像素级特征聚合，作为实例检测的输入) ==========
        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        # ========== CLIP 文本编码器 ==========
        self.clip_text_encoder = CLIPTextEncoder(
            clip_model=clip_model,
            embed_dim=512,
            freeze=True,
            pretrained_path=clip_weights_path
        )

        self.text_prompts = text_prompts or [
            "background, no change",
            "undamaged aircraft",
            "damaged aircraft",
            "undamaged building",
            "damaged building",
            "undamaged vehicle",
            "damaged vehicle",
        ]

        # ========== 文本-视觉交叉注意力 ==========
        self.cross_attn = TextVisualCrossAttention(
            visual_dim=128, text_dim=512, num_heads=8, dropout=0.1
        )

        # ========== 实例级检测头 ==========
        self.instance_head = InstanceDetectionHead(
            visual_dim=128,
            num_queries=num_queries,
            num_classes=output_classes,
            num_decoder_layers=6,
            nhead=8
        )

        # 保存类别数供损失函数使用
        self.output_classes = output_classes

    def forward(self, pre_data, post_data):
        """
        Args:
            pre_data:  (B, 3, H, W) 前时相图像
            post_data: (B, 3, H, W) 后时相图像

        Returns:
            dict:
                pred_boxes:     (B, num_queries, 4)
                pred_logits:    (B, num_queries, num_classes+1)
                instance_feats: (B, num_queries, 128)
        """
        # 1. 编码
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # 2. 像素级特征聚合 (ChangeDecoder 输出 128 通道特征图)
        pixel_features = self.decoder(pre_features, post_features)  # (B, 128, H/4, W/4)

        # 3. CLIP 文本编码
        text_features = self.clip_text_encoder(self.text_prompts)  # (N, 512)

        # 4. 文本-视觉交叉注意力增强
        enhanced_features = self.cross_attn(pixel_features, text_features)  # (B, 128, H/4, W/4)

        # 5. 实例级检测
        pred_boxes, pred_logits, instance_feats = self.instance_head(
            enhanced_features, text_features
        )

        return {
            'pred_boxes': pred_boxes,          # (B, num_queries, 4)
            'pred_logits': pred_logits,         # (B, num_queries, num_classes+1)
            'instance_feats': instance_feats,   # (B, num_queries, 128)
        }
