# HICD v6 架构设计文档

## 1. 设计动机

### 1.1 v5存在的问题

1. **Mamba backbone增益不明显**
   - VMamba/VSSM在变化检测任务上相比传统CNN/Transformer没有显著优势
   - Mamba擅长长序列建模，但遥感图像的空间关系更重要

2. **梯度冲突问题**
   - 共享backbone + 双分支（实例/语义）导致梯度相互干扰
   - 实例分支mAP接近0，其噪声梯度（权重3x）可能损害语义分支学习

3. **CLIP计算浪费**
   - CLIP主要用于state分类（区分损坏程度）
   - 但当前实现对target也有计算开销
   - 冻结策略下参数量仍然很大

4. **训练效率低**
   - 9168样本/batch_size=2 = 4584次迭代/epoch
   - 每个epoch需要10+小时
   - 显存占用高（24GB TITAN RTX几乎满载）

### 1.2 v6目标

- **更强的backbone**：DINOv3，密集特征提取能力更强
- **解决梯度冲突**：分离特征提取路径
- **优化CLIP**：只用于state，轻量化
- **提高训练效率**：LoRA微调，减少显存占用

---

## 2. 整体架构

```
输入图像对 (T1, T2)
    ↓
┌─────────────────────────────────────┐
│         DINOv3 Backbone (ViT-L)     │
│  - 冻结前80%层                       │
│  - 后20%层 + LoRA (rank=16)         │
│  - 输出多尺度特征: {C2, C3, C4, C5} │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│         Feature Difference Module   │
│  - 时相特征差分                      │
│  - 时相特征拼接 + 1x1 Conv          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Task-Specific Adapters (轻量)    │
│  ┌──────────────┐  ┌──────────────┐ │
│  │ Instance     │  │ Semantic     │ │
│  │ Adapter      │  │ Adapter      │ │
│  │ (MLP+LN)     │  │ (MLP+LN)     │ │
│  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────┘
    ↓              ↓
┌─────────────┐  ┌─────────────────┐
│  Instance   │  │    Semantic     │
│  Branch     │  │    Branch       │
│  ↓          │  │    ↓            │
│  DETR Head  │  │    FPN + 2 Heads│
│  (检测框)   │  │    (目标+状态)  │
└─────────────┘  └─────────────────┘
    ↓                    ↓
Instance Loss       CE + Dice Loss
    ↓                    ↓
┌─────────────────────────────────────┐
│        Weighted Joint Loss          │
│  L = λ1 * L_instance + λ2 * L_sem   │
│  λ1=1, λ2=1 (可学习)               │
└─────────────────────────────────────┘
```

---

## 3. 各模块详细设计

### 3.1 DINOv3 Backbone

**选型理由**：
- DINOv3 (ViT-L/14) 在语义分割、目标检测等密集任务上SOTA
- 原生支持多尺度特征提取
- 自监督预训练，语义理解能力强
- 社区支持好，HuggingFace官方实现

**实现方案**：

```python
import torch
import torch.nn as nn
from transformers import Dinov3Model, Dinov3Config

class DINOv3Backbone(nn.Module):
    def __init__(self, model_name="facebook/dinov3-vitl14", pretrained=True):
        super().__init__()
        self.dinov3 = Dinov3Model.from_pretrained(model_name)
        
        # 冻结前80%层
        total_layers = len(self.dinov3.encoder.layer)
        freeze_layers = int(total_layers * 0.8)
        for i, layer in enumerate(self.dinov3.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # LoRA微调后20%层
        self._apply_lora(rank=16)
        
        # 多尺度特征提取
        self.feature_pyramid = FeaturePyramidNetwork(
            in_channels=[1024, 1024, 1024, 1024],
            out_channels=256
        )
    
    def _apply_lora(self, rank=16):
        """在后20%层的Attention上应用LoRA"""
        total_layers = len(self.dinov3.encoder.layer)
        start_layer = int(total_layers * 0.8)
        
        for i in range(start_layer, total_layers):
            layer = self.dinov3.encoder.layer[i]
            # 在QKV投影上加LoRA
            layer.attention.attention.query = LoRALinear(
                layer.attention.attention.query, rank=rank
            )
            layer.attention.attention.key = LoRALinear(
                layer.attention.attention.key, rank=rank
            )
            layer.attention.attention.value = LoRALinear(
                layer.attention.attention.value, rank=rank
            )
    
    def forward(self, x):
        # DINOv3输出: last_hidden_state, hidden_states
        outputs = self.dinov3(x, output_hidden_states=True)
        
        # 提取多尺度特征
        # stride 4, 8, 16, 32
        features = {
            'C2': outputs.hidden_states[3],   # stride 4
            'C3': outputs.hidden_states[6],   # stride 8
            'C4': outputs.hidden_states[9],   # stride 16
            'C5': outputs.hidden_states[12],  # stride 32
        }
        
        return self.feature_pyramid(features)


class LoRALinear(nn.Module):
    """LoRA适配层"""
    def __init__(self, original_layer, rank=16, alpha=32):
        super().__init__()
        self.original = original_layer
        self.lora_a = nn.Linear(original_layer.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, original_layer.out_features, bias=False)
        self.alpha = alpha
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_a.weight)
        nn.init.zeros_(self.lora_b.weight)
    
    def forward(self, x):
        original_out = self.original(x)
        lora_out = self.lora_b(self.lora_a(x)) * (self.alpha / self.lora_a.weight.shape[0])
        return original_out + lora_out
```

**参数量分析**：
- DINOv3 ViT-L/14: ~304M参数
- 冻结前80%: ~243M冻结
- LoRA (rank=16): ~2M可训练
- 总可训练backbone参数: ~63M

### 3.2 Feature Difference Module

**目的**：捕捉时相变化

```python
class FeatureDifferenceModule(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        # 方案1: 特征差分
        self.diff_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        # 方案2: 特征拼接 + 1x1卷积
        self.concat_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
    
    def forward(self, feat_t1, feat_t2):
        # 差分特征
        diff = torch.abs(feat_t1 - feat_t2)
        diff = self.diff_conv(diff)
        
        # 拼接特征
        concat = torch.cat([feat_t1, feat_t2], dim=1)
        concat = self.concat_conv(concat)
        
        # 融合
        return diff + concat  # 残差连接
```

### 3.3 Task-Specific Adapters

**目的**：隔离实例和语义分支的梯度

```python
class TaskSpecificAdapter(nn.Module):
    def __init__(self, in_channels=256, adapter_channels=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_channels, adapter_channels),
            nn.LayerNorm(adapter_channels),
            nn.GELU(),
            nn.Linear(adapter_channels, in_channels),
            nn.LayerNorm(in_channels)
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.1)  # 可学习缩放
    
    def forward(self, x):
        return x + self.scale * self.adapter(x)  # 残差连接，小scale初始化
```

**设计要点**：
- 轻量级MLP + LayerNorm
- 小scale初始化（0.1），避免初始阶段影响主特征
- 残差连接保持特征稳定性
- 实例和语义分支各有独立adapter

### 3.4 Instance Branch

**基于DETR的目标检测头**

```python
class InstanceBranch(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=256):
        super().__init__()
        # DETR decoder
        self.decoder = DETRDecoder(
            d_model=hidden_dim,
            nhead=8,
            num_layers=6
        )
        
        # 预测头
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # x1, y1, x2, y2
        )
    
    def forward(self, features, queries):
        # features: FPN输出
        # queries: 可学习的object queries
        decoder_out = self.decoder(features, queries)
        
        # 预测
        class_logits = self.class_head(decoder_out)
        bbox_pred = self.bbox_head(decoder_out)
        
        return class_logits, bbox_pred
```

**输出**：
- target类别 (Building/Playground等)
- 边界框坐标 (x1, y1, x2, y2)
- **不使用CLIP**（target是视觉任务，不需要语义）

### 3.5 Semantic Branch

**轻量FPN + 双头**

```python
class SemanticBranch(nn.Module):
    def __init__(self, num_targets=4, num_states=4, in_channels=256):
        super().__init__()
        # FPN融合
        self.fpn = FPN(
            in_channels_list=[256, 256, 256, 256],
            out_channels=256
        )
        
        # 目标区域头 (前景/背景)
        self.target_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_targets, 1)
        )
        
        # 状态分类头 (损坏程度)
        self.state_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_states, 1)
        )
        
        # CLIP状态嵌入 (可选)
        self.use_clip = True
        if self.use_clip:
            self.clip_proj = nn.Linear(512, 256)  # CLIP text -> visual
    
    def forward(self, features, clip_text_features=None):
        # FPN融合
        fused = self.fpn(features)
        
        # 目标预测
        target_logits = self.target_head(fused)
        
        # 状态预测 (可选CLIP增强)
        state_logits = self.state_head(fused)
        
        if self.use_clip and clip_text_features is not None:
            # CLIP文本特征投影
            clip_proj = self.clip_proj(clip_text_features)
            # 余弦相似度增强
            state_logits = state_logits + cosine_similarity(fused, clip_proj)
        
        return target_logits, state_logits
```

**输出**：
- 目标区域图 (哪些像素属于变化目标)
- 状态图 (每个像素的损坏程度)
- **使用CLIP**（state是语义任务，需要语言指导）

### 3.6 CLIP模块（轻量化）

**设计原则**：只用于state，不解冻

```python
class LightweightCLIP(nn.Module):
    def __init__(self, clip_model_path):
        super().__init__()
        # 冻结整个CLIP
        self.clip = CLIPModel.from_pretrained(clip_model_path)
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # 只用text encoder
        self.text_encoder = self.clip.text_model
        
        # 轻量投影层
        self.text_proj = nn.Linear(512, 256)
    
    def forward(self, state_prompts):
        """
        state_prompts: ["no-damage", "minor-damage", ...]
        返回: state文本特征 [num_states, 256]
        """
        with torch.no_grad():
            text_features = self.text_encoder(state_prompts).last_hidden_state[:, 0, :]
        return self.text_proj(text_features)
```

**改进**：
- 完全冻结CLIP，无可训练参数
- 只用text encoder，不用vision encoder
- 只在semantic branch的state head使用
- 不参与target分类

### 3.7 损失函数

```python
class HICDv6Loss(nn.Module):
    def __init__(self, lambda_inst=1.0, lambda_sem=1.0):
        super().__init__()
        # 实例损失 (DETR style)
        self.instance_loss = DETRHungarianMatcherLoss(
            class_weight=1.0,
            bbox_weight=5.0,
            giou_weight=2.0
        )
        
        # 语义损失
        self.target_loss = nn.CrossEntropyLoss()
        self.state_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        
        # 权重 (可学习)
        self.lambda_inst = nn.Parameter(torch.tensor(lambda_inst))
        self.lambda_sem = nn.Parameter(torch.tensor(lambda_sem))
    
    def forward(self, predictions, targets):
        # 实例损失
        loss_inst = self.instance_loss(
            predictions['instance_logits'],
            predictions['instance_bbox'],
            targets['instance_gt']
        )
        
        # 语义损失
        loss_target = self.target_loss(
            predictions['target_logits'],
            targets['target_gt']
        )
        loss_state = self.state_loss(
            predictions['state_logits'],
            targets['state_gt']
        )
        loss_dice = self.dice_loss(
            predictions['target_logits'],
            targets['target_gt']
        )
        
        loss_sem = loss_target + loss_state + loss_dice
        
        # 加权求和
        total_loss = self.lambda_inst * loss_inst + self.lambda_sem * loss_sem
        
        return total_loss, {
            'loss_inst': loss_inst.item(),
            'loss_sem': loss_sem.item(),
            'loss_target': loss_target.item(),
            'loss_state': loss_state.item(),
            'loss_dice': loss_dice.item()
        }
```

---

## 4. 与v5对比

| 特性 | v5 | v6 |
|------|----|----|
| Backbone | VMamba (VSSM-Tiny) | DINOv3 (ViT-L/14) |
| 参数量 | ~30M | ~304M (冻结后~63M可训练) |
| 微调策略 | 全部解冻 | LoRA (rank=16) + 冻结前80% |
| CLIP | 冻结，用于target+state | 冻结，只用于state |
| 梯度隔离 | 无 | Task-Specific Adapters |
| 特征差分 | 简单差分 | 差分+拼接融合 |
| 检测头 | 自定义InstanceHead | DETR Decoder |
| 分割头 | FPN + Conv | FPN + Conv (同v5) |
| 训练效率 | 慢 (10+小时/epoch) | 预估3-4小时/epoch |

---

## 5. 训练策略

### 5.1 三阶段训练

**Stage 1: 冻结backbone (Epoch 1-10)**
- 冻结DINOv3全部参数
- 只训练Task-Specific Adapters + 检测/分割头
- 学习率: 1e-3
- 目标: 让任务头适配DINOv3特征

**Stage 2: LoRA解冻 (Epoch 11-50)**
- 解冻后20%层 + LoRA
- 学习率: 1e-4
- 目标: 让backbone适应变化检测任务

**Stage 3: 全模型微调 (Epoch 51-200)**
- 解冻全部层（保持LoRA）
- 学习率: 1e-5
- 目标: 精细调优

### 5.2 显存优化

- **梯度检查点**：DINOv3后20%层使用gradient checkpointing
- **混合精度**：FP16训练
- **梯度累积**：batch_size=2, grad_accum=8 (有效batch=16)
- **LoRA**：减少可训练参数，降低显存占用

### 5.3 预估训练时间

- 样本数: ~9000 (XBD)
- batch_size: 2, grad_accum: 8
- 每epoch迭代: 9000 / 2 / 8 = 562次
- 预估每iter: 3-5秒（DINOv3比Mamba快）
- 每epoch: 28-47分钟
- 200 epochs: 93-157小时 (4-6.5天)

**对比v5**：
- v5: 9168 / 2 = 4584次/epoch, 10+小时/epoch
- v6: 9000 / 2 / 8 = 562次/epoch, 0.5-1小时/epoch
- **加速10-20倍**

---

## 6. 实现计划

### 6.1 代码结构

```
HICD_v6/
├── models/
│   ├── backbone/
│   │   ├── dinov3_backbone.py      # DINOv3 + LoRA
│   │   └── feature_pyramid.py      # FPN
│   ├── modules/
│   │   ├── feature_diff.py         # 特征差分模块
│   │   ├── task_adapter.py         # Task-Specific Adapters
│   │   └── lightweight_clip.py     # 轻量CLIP
│   ├── heads/
│   │   ├── instance_head.py        # DETR实例头
│   │   └── semantic_head.py        # 语义分割头
│   ├── losses/
│   │   ├── detr_loss.py            # DETR损失
│   │   └── dual_branch_loss.py     # 双分支联合损失
│   └── HICD_v6.py                  # 主模型
├── datasets/
│   ├── dataset_v6.py               # 数据加载
│   └── configs/
│       ├── 0617final.yaml
│       ├── second.yaml
│       └── xbd.yaml
├── script/
│   └── train_full_v6.py            # 训练脚本
└── README.md
```

### 6.2 关键实现点

1. **DINOv3加载**：使用HuggingFace transformers库
2. **LoRA实现**：自定义LoRALinear层，支持任意Linear层
3. **DETR Decoder**：使用torch.nn.TransformerDecoder
4. **多尺度特征**：ViT输出需要reshape成2D特征图
5. **CLIP冻结**：完全冻结，只用text encoder

### 6.3 风险与对策

| 风险 | 对策 |
|------|------|
| DINOv3显存过大 | LoRA + 梯度检查点 + 冻结前80% |
| ViT特征尺度问题 | FPN多尺度融合 |
| DETR训练不稳定 | Hungarian Matcher + 学习率warmup |
| CLIP文本编码慢 | 预计算并缓存 |

---

## 7. 预期改进

### 7.1 性能提升

- **语义分割**：DINOv3特征更强，预计IoU提升5-10%
- **实例检测**：DETR decoder更稳定，mAP预计提升10-20%
- **训练效率**：LoRA + 梯度累积，时间减少10-20倍

### 7.2 工程优势

- **显存友好**：LoRA减少可训练参数，24GB显存足够
- **社区支持**：DINOv3 + HuggingFace生态，代码可维护性高
- **可扩展性**：Task-Specific Adapters易于添加新任务

---

## 8. 待讨论问题

1. **DINOv3版本选择**：ViT-L/14 vs ViT-B/14？
   - ViT-L更强但更大，ViT-B更轻量
   - 建议先用ViT-B验证，再升级ViT-L

2. **是否保留Mamba模块**？
   - 完全替换 vs Mamba作为补充模块
   - 建议完全替换，简化架构

3. **实例分支检测器**：
   - DETR vs RT-DETR vs YOLO
   - DETR更优雅但训练慢，YOLO更快
   - 建议先用DETR，如果太慢再换

4. **CLIP版本**：
   - ViT-B/32 vs ViT-L/14
   - 建议用ViT-B/32，够用且快

---

**作者**: HICD Team  
**日期**: 2026-08-04  
**版本**: v6 draft
