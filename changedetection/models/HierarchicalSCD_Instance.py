# -*- coding: utf-8 -*-
"""
HierarchicalSCDInstance — 层级实例级语义变化检测完整模型

数据流:
  pre_data  (B, 3, 512, 512)  ──┐
  post_data (B, 3, 512, 512) ──┤
                                ├──> Siamese VSSM Encoder
                                │    pre_features: 4 级特征
                                │    post_features: 4 级特征
                                │
                                ├──> ChangeDecoder
                                │    → pixel_features (B, 128, 128, 128)
                                │
                                ├──> CLIP Text Encoder
                                │    → text_features (16, 512)
                                │
                                ├──> TextVisualCrossAttention
                                │    → enhanced_features (B, 128, 128, 128)
                                │
                                └──> HierarchicalInstanceHead
                                     → pred_boxes   (B, Q, 4)
                                     → pred_target  (B, Q, 16)
                                     → pred_state   (B, Q, 6)
                                     → query_feats  (B, Q, 128)
                                     → aux_outputs  (训练时)
"""
import torch
import torch.nn as nn

from HICD.changedetection.models.Mamba_backbone import Backbone_VSSM
from HICD.classification.models.vmamba import LayerNorm2d
from HICD.changedetection.models.ChangeDecoder import ChangeDecoder
from HICD.changedetection.models.CLIPTextEncoder import CLIPTextEncoder
from HICD.changedetection.models.CrossAttentionFusion import TextVisualCrossAttention
from HICD.changedetection.models.HierarchicalInstanceHead import (
    HierarchicalInstanceHead,
)
from HICD.changedetection.models.class_mapping import (
    DatasetConfig,
    TARGET_NAMES, STATE_NAMES, CLIP_TEXT_PROMPTS,
    NUM_TARGETS, NUM_STATES,
)


class HierarchicalSCDInstance(nn.Module):
    """
    层级实例级语义变化检测模型

    核心创新:
      1. Mamba (VSSM) 骨干网络 — 高效长距离建模
      2. CLIP 文本引导 — 零样本/可扩展类别
      3. 层级检测头 — 目标类型 → 变化状态
      4. 多尺度 FPN — 处理极端尺度差异
      5. 辅助损失 — 加速训练收敛

    输入:
      pre_data:  (B, 3, H, W) 前时相影像
      post_data: (B, 3, H, W) 后时相影像

    输出:
      dict:
        pred_boxes:   (B, Q, 4)   归一化 bbox [cx, cy, w, h]
        pred_target:  (B, Q, 16)  目标类型 logits
        pred_state:   (B, Q, 6)   变化状态 logits (有效性掩码)
        query_feats:  (B, Q, 128) 实例特征
        aux_outputs:  list[dict]   辅助层预测 (训练时)
    """
    def __init__(self, pretrained, num_queries_per_scale=34, dataset_config=None, clip_mode='both',
                 clip_model="ViT-B-16", clip_weights_path=None,
                 **kwargs):
        super().__init__()
        self.dataset_config = dataset_config
        self.clip_mode = clip_mode

        # ── 1. Siamese VSSM Encoder ──
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

        # ── 2. ChangeDecoder ──
        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        # ── 3. CLIP Text Encoder ──
        self.clip_text_encoder = CLIPTextEncoder(
            clip_model=clip_model, embed_dim=128,
            freeze=True, pretrained_path=clip_weights_path
        )

        # ── 4. Cross-Attention (文本 → 视觉) ──
        self.cross_attn = TextVisualCrossAttention(
            visual_dim=128, text_dim=128, num_heads=8, dropout=0.1
        )

        # ── 5. Hierarchical Instance Detection Head ──
        self.instance_head = HierarchicalInstanceHead(
            visual_dim=128,
            num_queries_per_scale=num_queries_per_scale,
            dataset_config=dataset_config,
            num_targets=dataset_config.num_targets if dataset_config else NUM_TARGETS,
            num_states=dataset_config.num_states if dataset_config else NUM_STATES,
            num_decoder_layers=6,
            nhead=8,
        )

    def forward(self, pre_data, post_data):
        """
        Args:
            pre_data:  (B, 3, H, W) 前时相影像
            post_data: (B, 3, H, W) 后时相影像

        Returns:
            dict: 见类文档
        """
        # 1. 编码 (共享权重)
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # 2. 变化检测特征融合
        # 2. 变化检测特征融合 (V2: multi-scale)
        pixel_features = self.decoder(pre_features, post_features)  # (p1, p2, p3)
        p1, p2, p3 = pixel_features

        # 3. CLIP 文本编码
        text_features = self.clip_text_encoder(self.dataset_config.clip_text_prompts)

        # 4. 文本-视觉交叉注意力增强
        # 4. 文本-视觉交叉注意力增强 (V2: apply to each scale)
        enhanced_p1 = self.cross_attn(p1, text_features)
        enhanced_p2 = self.cross_attn(p2, text_features)
        enhanced_p3 = self.cross_attn(p3, text_features)

        # 5. 实例检测
        # 5. 实例检测 (V2: multi-scale)
        outputs = self.instance_head(
            (enhanced_p1, enhanced_p2, enhanced_p3),
            text_features, multi_scale=True
        )

        return outputs

    @torch.no_grad()
    def inference(self, pre_data, post_data, confidence_threshold=0.3):
        """
        推理接口: 返回过滤后的检测结果

        Args:
            pre_data:  (B, 3, H, W)
            post_data: (B, 3, H, W)
            confidence_threshold: 目标类型置信度阈值

        Returns:
            list of dict (每个样本):
                boxes:   (K, 4) 过滤后的 bbox
                targets: (K,)   目标类型索引
                states:  (K,)   变化状态索引
                scores:  (K,)   置信度
        """
        # Forward with state CLIP if available
        outputs = self.forward(pre_data, post_data)
        pred_target = outputs['pred_target']   # (B, Q, 16)
        pred_state = outputs['pred_state']     # (B, Q, 6)
        pred_boxes = outputs['pred_boxes']     # (B, Q, 4)

        B = pred_target.shape[0]
        results = []

        for b in range(B):
            target_probs = torch.softmax(pred_target[b], dim=-1)  # (Q, 16)
            max_target_prob, target_indices = target_probs.max(dim=-1)  # (Q,)

            state_probs = torch.softmax(pred_state[b], dim=-1)  # (Q, 6)
            max_state_prob, state_indices = state_probs.max(dim=-1)  # (Q,)

            # 过滤低置信度
            valid = max_target_prob > confidence_threshold

            results.append({
                'boxes': pred_boxes[b][valid],
                'targets': target_indices[valid],
                'states': state_indices[valid],
                'scores': max_target_prob[valid],
            })

        # V3: Crater-aware state propagation
        results = self._propagate_damage_state(results)

        return results

    # ── Crater-aware state propagation ────────────────────────────
    # 大型基础设施(跑道/滑行道/停机坪)的损坏往往只占很小面积,
    # 被弹坑覆盖后应自动升级为 Damaged 状态。
    # 与人类分析师思路一致: 先检测弹坑, 再推断基础设施受损。
    _INFRA_TARGETS = {0, 1, 2, 3}  # Farmland, Runway, Taxiway, Apron
    _CRATER_IDX = 9              # Crater (new idx)

    def _propagate_damage_state(self, results):
        """
        对每个样本: 如果 Crater 的中心点落在 Runway/Taxiway/Apron 的 bbox 内,
        将该基础设施状态升级为 Damaged(1)。

        逻辑: 弹坑是点状目标, 跑道是面状目标。
        判断弹坑是否"在跑道内"用中心点包含, 不用 IoU。
        与人类分析师一致: 看到弹坑在跑道上 → 跑道不可用。
        """
        for res in results:
            boxes = res['boxes']      # (N, 4) [cx, cy, w, h]
            targets = res['targets']  # (N,)
            states = res['states']    # (N,)
            n = len(boxes)
            if n == 0:
                continue

            # 找出所有 Crater 的中心点
            crater_mask = (targets == self._CRATER_IDX)
            crater_centers = boxes[crater_mask, :2]  # (C, 2) [cx, cy]

            if len(crater_centers) == 0:
                continue

            # 对每个基础设施实例, 检查弹坑中心是否在其 bbox 内
            for i in range(n):
                if targets[i].item() not in self._INFRA_TARGETS:
                    continue
                if states[i].item() == 1:  # 已经是 Damaged
                    continue

                infra_cx, infra_cy = boxes[i, 0].item(), boxes[i, 1].item()
                infra_hw = boxes[i, 2].item() / 2  # half width
                infra_hh = boxes[i, 3].item() / 2  # half height
                x1, y1 = infra_cx - infra_hw, infra_cy - infra_hh
                x2, y2 = infra_cx + infra_hw, infra_cy + infra_hh

                # 检查任一弹坑中心是否在 bbox 内
                inside_x = (crater_centers[:, 0] >= x1) & (crater_centers[:, 0] <= x2)
                inside_y = (crater_centers[:, 1] >= y1) & (crater_centers[:, 1] <= y2)
                if (inside_x & inside_y).any():
                    states[i] = 1  # 升级为 Damaged






