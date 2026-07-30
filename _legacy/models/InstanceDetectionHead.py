import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceDetectionHead(nn.Module):
    """
    DETR 风格的实例检测头。
    
    核心思想：用可学习的 object queries 查询像素级特征，
    每个 query 聚合一个实例的信息，输出 bbox + 损毁类别。
    """
    def __init__(self, visual_dim=128, num_queries=100, num_classes=7,
                 num_decoder_layers=6, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # 可学习的 object queries
        self.query_embed = nn.Embedding(num_queries, visual_dim)
        
        # 查询位置编码（可学习的参考点）
        self.query_pos = nn.Embedding(num_queries, visual_dim)
        
        # Transformer 解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=visual_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # bbox 输出头: (x_center, y_center, width, height) 归一化坐标
        self.bbox_head = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, 4),
            nn.Sigmoid()
        )
        
        # 分类头: 每个 query 的损毁类别
        self.class_head = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.ReLU(),
            nn.Linear(visual_dim, num_classes + 1)  # +1 for "no object"
        )
        
        self.instance_norm = nn.LayerNorm(visual_dim)
    
    def forward(self, pixel_features, text_features=None):
        """
        Args:
            pixel_features: (B, C, H, W) - 来自 ChangeDecoder 的像素级特征
            text_features: (N, text_dim) - 可选，文本语义特征用于增强分类
        Returns:
            pred_boxes: (B, num_queries, 4)
            pred_logits: (B, num_queries, num_classes+1)
            query_feats: (B, num_queries, C)
        """
        B, C, H, W = pixel_features.shape
        
        # 像素特征展平为序列: (B, HW, C)
        memory = pixel_features.flatten(2).permute(0, 2, 1)
        
        # 位置编码
        pos = self._get_sincos_pos_embed(H, W, C).unsqueeze(0).expand(B, -1, -1)
        memory = memory + pos.to(memory.device)
        
        # Object queries: (1, num_queries, C) -> (B, num_queries, C)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Transformer 解码
        instance_feats = self.decoder(
            tgt=queries + query_pos,
            memory=memory
        )  # (B, num_queries, C)
        
        instance_feats = self.instance_norm(instance_feats)
        
        # 预测 bbox 和类别
        pred_boxes = self.bbox_head(instance_feats)  # (B, num_queries, 4)
        pred_logits = self.class_head(instance_feats)  # (B, num_queries, num_classes+1)
        
        # 如果有文本特征，用余弦相似度增强分类 logits
        if text_features is not None:
            inst_norm = F.normalize(instance_feats, dim=-1)  # (B, Q, C)
            txt_norm = F.normalize(text_features, dim=-1)  # (N, C)
            cosine_sim = torch.matmul(inst_norm, txt_norm.t())  # (B, Q, N)
            pred_logits[:, :, :self.num_classes] = (
                pred_logits[:, :, :self.num_classes] + cosine_sim
            ) / 2
        
        return pred_boxes, pred_logits, instance_feats
    
    def _get_sincos_pos_embed(self, H, W, C):
        """生成 2D 正弦余弦位置编码"""
        pe = torch.zeros(H * W, C)
        y_pos = torch.arange(H).unsqueeze(1).expand(H, W).flatten().float()
        x_pos = torch.arange(W).unsqueeze(0).expand(H, W).flatten().float()
        
        div_term = torch.exp(
            torch.arange(0, C // 2, 2).float() * -(math.log(10000.0) / (C // 2))
        )
        
        pe[:, 0::4] = torch.sin(x_pos.unsqueeze(1) * div_term)
        pe[:, 1::4] = torch.cos(x_pos.unsqueeze(1) * div_term)
        pe[:, 2::4] = torch.sin(y_pos.unsqueeze(1) * div_term)
        pe[:, 3::4] = torch.cos(y_pos.unsqueeze(1) * div_term)
        
        return pe
