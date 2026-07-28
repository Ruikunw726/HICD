# MambaCD — 层级实例级语义变化检测

基于 Mamba (VSSM) + CLIP 文本引导的多目标多类别实例级遥感变化检测框架。

## 核心创新

1. **Mamba 骨干网络** — 用 State Space Model 替代 Transformer 做遥感变化检测, 线性复杂度处理高分辨率影像
2. **CLIP 文本引导** — 利用视觉-语言模型的语义先验, 支持零样本/可扩展类别
3. **层级检测头** — 目标类型 → 变化状态的两层分类, 通过有效性矩阵约束合法组合
4. **多尺度 FPN** — 自顶向下路径 + 侧向连接, 处理 5000 倍尺度差异
5. **辅助层损失** — 中间 decoder 层监督, 加速收敛

## 数据集

**0617final** — 9,687 对 512×512 VHR 双时相影像

| 场景 | 图片数 | 实例数 | 主要目标 |
|------|--------|--------|---------|
| Airports | 1,101 | 5,958 | 跑道/滑行道/停机坪/建筑 |
| Ports | 596 | 4,868 | 栈桥/船坞/舰船 |
| Urban-Rural Areas | 7,990 | 204,860 | 建筑物/公路/农田 |

**类别体系** (16 目标 × 6 状态, 68 非背景类):

| 目标类型 | 状态 | 特点 |
|---------|------|------|
| Farmland | NoChange, Damaged | 仅 2 种状态 |
| Runway/Taxiway/Apron/Bridge/Highway | NoChange, Damaged, Reduced, Added, Extended | 5 种状态 |
| Building/Shelter/Tower/Pier/Dock | NoChange, Damaged, Reduced, Added, Extended | 5 种状态 |
| Tank | NoChange, Damaged, Reduced, Added | 4 种状态 (无 Extended) |
| Aircraft/Vessel | NoChange, Damaged, Reduced, Added, Replaced | 5 种状态 (无 Extended, 有 Replaced) |
| Crater/VehicleRevet | 无状态 | Stateless |

## 模型架构

```
输入:
  pre_data  (B, 3, 512, 512)  前时相影像
  post_data (B, 3, 512, 512)  后时相影像
      │
      ▼
┌─────────────────────────────────────────┐
│  Siamese VSSM Encoder (共享权重)         │
│  ┌─────────────────────────────────┐    │
│  │ Backbone_VSSM                   │    │
│  │ out_indices=(0,1,2,3)           │    │
│  │ → 4 级特征 × 2                  │    │
│  └─────────────────────────────────┘    │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  ChangeDecoder                          │
│  ┌─────────────────────────────────┐    │
│  │ 4 级 VSSBlock 时序融合           │    │
│  │ 3 种特征拼接方式:                │    │
│  │   - cat([pre, post])            │    │
│  │   - 交错列拼接                  │    │
│  │   - 左右拼接                    │    │
│  │ + 自顶向下 FPN 融合             │    │
│  └─────────────────────────────────┘    │
│  输出: pixel_features (B, 128, 128, 128) │
└─────────────┬───────────────────────────┘
              │
              ├──→ ┌──────────────────┐
              │    │ CLIP Text Encoder │
              │    │ 16 个文本提示词   │
              │    │ → (16, 512)      │
              │    └────────┬─────────┘
              │             │
              ▼             ▼
┌─────────────────────────────────────────┐
│  TextVisualCrossAttention               │
│  ┌─────────────────────────────────┐    │
│  │ Query: 视觉特征                 │    │
│  │ Key/Value: 文本特征             │    │
│  │ + 门控融合 + 残差连接           │    │
│  └─────────────────────────────────┘    │
│  输出: enhanced (B, 128, 128, 128)      │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  HierarchicalInstanceHead               │
│                                         │
│  ┌─ ScaleFPN ─────────────────────┐    │
│  │ P3 (128, 128, 128) 小目标      │    │
│  │ P4 (128,  64,  64) 中目标      │    │
│  │ P5 (128,  32,  32) 大目标      │    │
│  └─────────────────────────────────┘    │
│           │                              │
│           ▼                              │
│  ┌─ ScaleAwareQueryEmbedding ─────┐    │
│  │ 34 queries × 3 scales = 102    │    │
│  └─────────────────────────────────┘    │
│           │                              │
│           ▼                              │
│  ┌─ Transformer Decoder ──────────┐    │
│  │ 6 层, 8 头                     │    │
│  │ 中间层 2, 4 输出辅助预测       │    │
│  └─────────────────────────────────┘    │
│           │                              │
│           ▼                              │
│  ┌─ 预测头 ───────────────────────┐    │
│  │ bbox_head:  → (B, Q, 4)        │    │
│  │ target_head: → (B, Q, 16)      │    │
│  │ state_head: → (B, Q, 6)        │    │
│  │ + target_state_mask 有效性约束  │    │
│  │ + CLIP cosine similarity 增强   │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘

输出:
  pred_boxes:   (B, 102, 4)    归一化 bbox [cx, cy, w, h]
  pred_target:  (B, 102, 16)   目标类型 logits
  pred_state:   (B, 102, 6)    变化状态 logits
  query_feats:  (B, 102, 128)  实例特征
  aux_outputs:  [dict×2]       辅助层预测 (训练时)
```

## 代码文件说明

### 模型 (`changedetection/models/`)

| 文件 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `class_mapping.py` | **类别定义唯一数据源** — 16 目标 × 6 状态映射, 有效性矩阵, CLIP 提示词 | — | TARGET_NAMES, STATE_NAMES, TARGET_VALID_STATES |
| `HierarchicalSCD_Instance.py` | **完整模型** — 串联所有组件 | pre/post (B,3,512,512) | pred_boxes/target/state |
| `HierarchicalInstanceHead.py` | **检测头** — FPN + Transformer Decoder + 分层预测 | pixel_features (B,128,H/4,W/4) | pred_boxes/target/state + aux |
| `HierarchicalInstanceLoss.py` | **损失函数** — Hungarian 匹配 + Focal + Dice + GIoU | 模型输出 + GT | loss, loss_dict |
| `Mamba_backbone.py` | VSSM 骨干网络 (Siamese) | (B,3,H,W) | 4 级特征 |
| `ChangeDecoder.py` | 变化检测特征融合 (3 种拼接 + VSSBlock) | 4 级 pre/post 特征 | (B,128,H/4,W/4) |
| `CLIPTextEncoder.py` | CLIP 文本编码器 (冻结) | 文本列表 | (N, 512) |
| `CrossAttentionFusion.py` | 文本-视觉交叉注意力 + 门控 | 视觉 (B,C,H,W) + 文本 (N,D) | (B,C,H,W) |

### 旧版模型 (保留, 不再使用)

| 文件 | 说明 |
|------|------|
| `MambaBCD.py` | 二值变化检测 (BCD) |
| `STMambaSCD.py` | 语义变化检测 (SCD), 像素级 |
| `STMambaBDA.py` | 建筑物损毁评估 (BDA) |
| `STMambaSCD_Instance.py` | 旧版实例检测头 |
| `InstanceDetectionHead.py` | 旧版检测头 |
| `InstanceLoss.py` | 旧版损失函数 |
| `SemanticDecoder.py` | 语义解码器 |

### 数据处理 (`changedetection/datasets/`)

| 文件 | 作用 |
|------|------|
| `pixel_to_instance_0617final.py` | **像素→实例转换** — 连通域分析, 输出 instances.json |
| `label_statistics.py` | 标签统计 — 每类数量/面积分布 |
| `imutils.py` | 图像工具 — 归一化, 数据增强 |
| `label_change.py` | 标签变化映射 |
| `make_data_loader.py` | 像素级数据加载器 |
| `make_data_loader_pure.py` | 纯数据加载器 (无依赖) |

### 训练/推理 (`changedetection/script/`)

| 文件 | 作用 |
|------|------|
| `train_InstanceSCD.py` | **实例级训练脚本** — 完整训练流程 |
| `train_MambaBCD.py` | BCD 训练 |
| `train_MambaSCD.py` | SCD 训练 |
| `train_MambaBDA.py` | BDA 训练 |
| `infer_MambaBCD.py` | BCD 推理 |
| `infer_MambaSCD.py` | SCD 推理 |
| `infer_MambaBDA.py` | BDA 推理 |

### 工具 (`changedetection/utils_func/`)

| 文件 | 作用 |
|------|------|
| `metrics.py` | 评估指标 |
| `eval_segm.py` | 语义分割评估 |
| `lovasz_loss.py` | Lovász 损失 |
| `mcd_utils.py` | 变化检测工具函数 |

## 模型 I/O 详解

### 训练时 I/O

```python
# 输入
pre_data:  Tensor (B, 3, 512, 512)  # 前时相 RGB, 归一化到 [0,1]
post_data: Tensor (B, 3, 512, 512)  # 后时相 RGB, 归一化到 [0,1]

# GT (来自 instances.json)
gt_boxes:   list[Tensor]  # 每个样本 (M_i, 4) 归一化 [cx,cy,w,h]
gt_target:  list[Tensor]  # 每个样本 (M_i,) 目标类型 0-15
gt_state:   list[Tensor]  # 每个样本 (M_i,) 变化状态 0-5

# 输出
outputs = model(pre_data, post_data)
# outputs['pred_boxes']:    (B, 102, 4)
# outputs['pred_target']:   (B, 102, 16)
# outputs['pred_state']:    (B, 102, 6)
# outputs['query_feats']:   (B, 102, 128)
# outputs['aux_outputs']:   [{'pred_boxes':..., 'pred_target':..., 'pred_state':...}, ...]

# 损失计算
loss, loss_dict = criterion(outputs, gt_boxes, gt_target, gt_state)
```

### 推理时 I/O

```python
# 输入 (同训练)
pre_data:  Tensor (B, 3, 512, 512)
post_data: Tensor (B, 3, 512, 512)

# 输出 (过滤后)
results = model.inference(pre_data, post_data, confidence_threshold=0.3)
# results: list[dict]
#   results[b]['boxes']:   (K, 4) 过滤后的 bbox
#   results[b]['targets']: (K,)   目标类型索引
#   results[b]['states']:  (K,)   变化状态索引
#   results[b]['scores']:  (K,)   置信度
```

### 数据流水线

```
原始影像 (GeoTIFF)
    │
    ├── pixel_to_instance_0617final.py
    │   └── 连通域分析 → instances.json
    │
    ├── train_InstanceSCD.py
    │   ├── 加载 instances.json
    │   ├── 读取 pre/post GeoTIFF
    │   ├── 数据增强 (翻转/旋转)
    │   ├── 归一化 + CHW 转换
    │   └── 送入模型
    │
    └── 推理
        ├── 加载 pre/post GeoTIFF
        ├── 归一化
        ├── 模型前向
        └── 后处理 (NMS/阈值过滤)
```

## 使用指南

### 1. 数据准备

```bash
# 为每个场景生成实例标注
python changedetection/datasets/pixel_to_instance_0617final.py \
    --data_dir D:\CD\0617final\Airports \
    --classes_csv D:\CD\0617final\classes.csv

# 批量处理所有场景
python changedetection/datasets/pixel_to_instance_0617final.py \
    --data_dir D:\CD\0617final\Airports \
    --all_scenes
```

### 2. 训练

```bash
python changedetection/script/train_InstanceSCD.py \
    --data_dir D:\CD\0617final\Airports \
    --instances_json D:\CD\0617final\Airports\instances.json \
    --pretrained_weight_path /path/to/vssm1_tiny_224.pth \
    --batch_size 2 \
    --crop_size 512 \
    --max_epochs 100 \
    --use_amp
```

### 3. 标签统计

```bash
python changedetection/datasets/label_statistics.py
```

## 类别映射表

| target_idx | Target | state_idx | State | train_id 范围 |
|-----------|--------|-----------|-------|--------------|
| 0 | Farmland | 0,1 | NoChange, Damaged | 1-2 |
| 1 | Runway | 0-4 | NoChange, Damaged, Reduced, Added, Extended | 3-7 |
| 2 | Taxiway | 0-4 | 同上 | 8-12 |
| 3 | Apron | 0-4 | 同上 | 13-17 |
| 4 | Bridge | 0-4 | 同上 | 18-22 |
| 5 | Highway | 0-4 | 同上 | 23-27 |
| 6 | Building | 0-4 | 同上 | 28-32 |
| 7 | Shelter | 0-4 | 同上 | 33-37 |
| 8 | Tower | 0-4 | 同上 | 38-42 |
| 9 | Pier | 0-4 | 同上 | 43-47 |
| 10 | Dock | 0-4 | 同上 | 48-52 |
| 11 | Tank | 0-3 | NoChange, Damaged, Reduced, Added | 53-56 |
| 12 | Aircraft | 0-3,5 | NoChange, Damaged, Reduced, Added, Replaced | 57-61 |
| 13 | Vessel | 0-3,5 | 同上 | 62-66 |
| 14 | Crater | 0 | 无状态 | 67 |
| 15 | VehicleRevet | 0 | 无状态 | 68 |
