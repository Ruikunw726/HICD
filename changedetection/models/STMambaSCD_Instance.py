import torch
import torch.nn as nn
import torch.nn.functional as F
from MambaCD.changedetection.models.Mamba_backbone import Backbone_VSSM
from MambaCD.classification.models.vmamba import LayerNorm2d
from MambaCD.changedetection.models.ChangeDecoder import ChangeDecoder
from MambaCD.changedetection.models.SemanticDecoder import SemanticDecoder
from MambaCD.changedetection.models.CLIPTextEncoder import CLIPTextEncoder
from MambaCD.changedetection.models.CrossAttentionFusion import TextVisualCrossAttention
from MambaCD.changedetection.models.InstanceDetectionHead import InstanceDetectionHead


class STMambaSCD_Instance(nn.Module):
    """
    在 STMambaSCD 基础上新增：
    1. CLIP 文本编码器 + 交叉注意力 -> 语义引导分类
    2. 实例级检测头 -> 输出 bbox + 损毁类别
    """
    def __init__(self, output_cd, output_clf, pretrained,
                 num_queries=100, clip_model="ViT-B/16",
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
        
        # ========== 解码器 (现有) ==========
        self.decoder_bcd = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )
        self.decoder_T1 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )
        self.decoder_T2 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )
        
        # ========== CLIP 文本编码器 (新增) ==========
        self.clip_text_encoder = CLIPTextEncoder(
            clip_model=clip_model,
            embed_dim=512,
            freeze=True,
            pretrained_path=clip_weights_path
        )
        
        # 文本提示词
        self.text_prompts = text_prompts or [
            "background, no change",
            "destroyed building",
            "partially damaged building",
            "intact building",
            "damaged road",
            "flooded area",
            "collapsed structure",
            "burned vegetation",
        ]
        
        # ========== 文本-视觉交叉注意力 (新增) ==========
        self.cross_attn = TextVisualCrossAttention(
            visual_dim=128, text_dim=512, num_heads=8, dropout=0.1
        )
        
        # ========== 实例级检测头 (新增) ==========
        self.instance_head = InstanceDetectionHead(
            visual_dim=128,
            num_queries=num_queries,
            num_classes=output_clf,
            num_decoder_layers=6,
            nhead=8
        )
        
        # ========== 分类头 ==========
        self.main_clf_cd = nn.Conv2d(128, output_cd, kernel_size=1)
        self.aux_clf = nn.Conv2d(128, output_clf, kernel_size=1)
    
    def forward(self, pre_data, post_data):
        # ====== 1. 编码 ======
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        
        # ====== 2. 像素级解码 ======
        output_bcd = self.decoder_bcd(pre_features, post_features)
        output_T1 = self.decoder_T1(pre_features)
        output_T2 = self.decoder_T2(post_features)
        
        # ====== 3. CLIP 文本编码 ======
        text_features = self.clip_text_encoder(self.text_prompts)  # (N, 512)
        
        # ====== 4. 文本-视觉交叉注意力融合 ======
        output_bcd_enhanced = self.cross_attn(output_bcd, text_features)
        
        # ====== 5. 实例级检测 ======
        pred_boxes, pred_logits, instance_feats = self.instance_head(
            output_bcd_enhanced, text_features
        )
        
        # ====== 6. 像素级分类 ======
        output_bcd_pixel = self.main_clf_cd(output_bcd_enhanced)
        output_bcd_pixel = F.interpolate(
            output_bcd_pixel, size=pre_data.shape[-2:], mode='bilinear'
        )
        
        output_T1 = self.aux_clf(output_T1)
        output_T1 = F.interpolate(
            output_T1, size=pre_data.shape[-2:], mode='bilinear'
        )
        
        output_T2 = self.aux_clf(output_T2)
        output_T2 = F.interpolate(
            output_T2, size=pre_data.shape[-2:], mode='bilinear'
        )
        
        return {
            'pixel_change_map': output_bcd_pixel,
            'pixel_T1_map': output_T1,
            'pixel_T2_map': output_T2,
            'pred_boxes': pred_boxes,
            'pred_logits': pred_logits,
            'instance_feats': instance_feats,
        }
