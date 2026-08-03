# HICD — Hierarchical Instance Change Detection

基于 Mamba (VSSM) + CLIP 文本引导的层级实例级遥感变化检测框架。

## 核心创新

1. **Mamba 骨干网络** — 用 State Space Model 替代 Transformer 做遥感变化检测, 线性复杂度处理高分辨率影像
2. **CLIP 文本引导** — 利用视觉-语言模型的语义先验, 支持零样本/可扩展类别 (V4.1: 两阶段训练, 前20 epoch冻结后解冻最后2层)
3. **层级检测头** — 目标类型 → 变化状态的两层分类, 通过有效性矩阵约束合法组合
4. **多尺度 FPN** — 自顶向下路径 + 侧向连接, 处理 5000 倍尺度差异
5. **辅助层损失** — 中间 decoder 层监督, 加速收敛
6. **One-to-Many 匹配** — V3: 每个 GT 匹配 Top-K 个 query, 正样本增加 3 倍, 收敛更快
7. **变化注意力 (Change Attention)** — V3: cross-attention 让 state 分类聚焦变化最剧烈区域, 解决大目标小损伤的信号淹没问题
8. **弹坑感知状态传播** — V3: 推理时弹坑中心点落在基础设施 bbox 内 → 自动升级为 Damaged
9. **ICD 统一评估协议** — 实例级评估框架, 公平对比像素级和实例级方法
10. **SD-SSM (Spatial Difference-aware SSM)** — V4: ChangeDecoder 每个 stage 新增差值分支, 显式将 pre-post 特征差送入 SSM, 让门控机制直接建模"哪里变了"
11. **Context-SSM** — V4: 多尺度深度可分离卷积注入局部空间上下文, 弥补 SSM 展平 2D→1D 时丢失的局部信息
12. **Pair-weighted State Loss** — V4: 对稀有 (target, state) 组合加权 (如 Aircraft/Vessel+Replaced 3x), 缓解实例级类别不平衡
13. **SparseChangeGate** — V4.2: SD-SSM 分支前加可学习软阈值门控, 受 SNN time-to-first-spike 启发, 抑制噪声差分、只让显著变化进入 SSM, 提升信噪比和收敛速度
14. **双分支解码器 (Dual-Branch Decoder)** — V5: 实例检测 + 语义分割并行, 小目标用 bbox、大范围目标用像素级分割, Task-Specific Adapters 避免梯度冲突
15. **DatasetConfig 分支路由** — V5: YAML 配置指定每个类别走哪个分支, 新数据集只需改配置文件 — V4.2: SD-SSM 分支前加可学习软阈值门控, 受 SNN time-to-first-spike 启发, 抑制噪声差分、只让显著变化进入 SSM, 提升信噪比和收敛速度

## 数据集

**0617final** — 9,687 对 512×512 VHR 双时相影像

| 场景 | Train | Val | Test | 实例数 |
|------|-------|-----|------|--------|
| Airports | 660 | 220 | 221 | 5,958 |
| Ports | 358 | 119 | 119 | 4,868 |
| Urban-Rural Areas | 4,937 | 1,234 | 1,819 | 204,860 |

**类别体系**: 10 目标类型 × 6 变化状态, 48 个非背景 train_id。详见 `0617final/classes.csv`。

## 项目结构

```
HICD/
├── 0617final/                 # 数据集 (含 instances.json)
├── weights/                   # 预训练权重
│   ├── vssm1_small_0229s_ckpt_epoch_240.pth  # VSSM-small backbone (V3 默认)
│   ├── vssmtiny_dp01_ckpt_epoch_292.pth      # VSSM-tiny backbone (V2)
│   └── open_clip_pytorch_model.bin           # CLIP 文本编码器
├── outputs/                   # 训练输出
├── classification/
│   └── models/vmamba.py       # VSSM 模型实现
├── changedetection/
│   ├── configs/vssm1/         # 模型配置 (YAML)
│   ├── datasets/
│   │   ├── pixel_to_instance_0617final.py  # 像素→实例标注转换
│   │   └── imutils.py                      # 图像工具
│   ├── models/
│   │   ├── HierarchicalSCD_Instance.py     # 主模型
│   │   ├── Mamba_backbone.py               # VSSM backbone (含预训练加载)
│   │   ├── ChangeDecoder.py                # 多尺度变化解码器
│   │   ├── CLIPTextEncoder.py              # CLIP 文本编码
│   │   ├── CrossAttentionFusion.py         # 文本-视觉交叉注意力
│   │   ├── HierarchicalInstanceHead.py     # 实例检测头
│   │   ├── HierarchicalInstanceLoss.py     # 层级损失函数
│   │   └── class_mapping.py                # 类别定义 (唯一数据源)
│   └── script/
│       ├── train_full.py       # 全量训练脚本 (多场景联合训练)
│       ├── train_InstanceSCD.py # 单场景训练脚本
│       ├── metrics.py          # 评估指标模块 (mAP/F1/速度/计算量)
│       └── eval_unified.py     # ICD 统一评估协议
├── kernels/selective_scan/    # CUDA 扩展
├── Dockerfile                 # Docker 部署
├── requirements.txt
└── TROUBLESHOOTING.md         # 踩坑记录
```

## 快速开始

### 环境配置

```bash
# 创建 conda 环境
conda create -n mamba python=3.11 -y && conda activate mamba

# 安装 PyTorch (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装依赖
pip install -r requirements.txt

# 安装 GDAL
pip install GDAL==$(gdal-config --version)

# 编译 CUDA 扩展
cd kernels/selective_scan && pip install -e . --no-build-isolation && cd ../..
```

### 数据准备

```bash
# 为每个场景生成实例标注 (instances.json)
python changedetection/datasets/pixel_to_instance_0617final.py \
    --data_dir 0617final/Airports --classes_csv 0617final/classes.csv
```

#

## V4.2 更新 (2026-07-31)

| 改动 | 内容 |
|------|------|
| SparseChangeGate | SD-SSM 分支新增可学习软阈值门控, 噪声差分被抑制、只保留显著变化进入 SSM |
| 灵感来源 | SpikeAdapter (CVPR 2026) 的 GSI-P: SNN time-to-first-spike 编码的稀疏激活思想 |
| 参数开销 | 每尺度 1 个可学习阈值, 共 4 个参数, 几乎为零 |
| 原理 | 等价于信号处理中的软阈值 (soft thresholding), 变化检测信号天然是稀疏的, 显式建模比隐式学习更高效 |

## V4.1 更新 (2026-07-31)

| 改动 | 内容 |
|------|------|
| VSSM tiny | backbone 从 small (15层, 50M) → tiny (4层, 30M), 减少计算量和显存 |
| CLIP 两阶段训练 | 前 20 epoch 冻结 CLIP, 之后解冻最后 2 层 (lr=0.1x), 避免早期梯度破坏语义先验 |
| 训练指令 | `--clip_unfreeze_epoch 20` (默认), `0` = 始终冻结, `-1` = 始终解冻 |
## 训练

```bash
cd /path/to/HICD/..
export PYTHONPATH="$(pwd):$PYTHONPATH"

python HICD/changedetection/script/train_full.py \
    --data_dir HICD/0617final \
    --scenes "Airports,Ports,Urban-Rural Areas" \
    --batch_size 4 --grad_accum 4 \
    --learning_rate 3e-4 --max_epochs 100 \
    --use_amp --exp_name experiment_1
```

**训练参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 4 | 每步样本数 |
| `--grad_accum` | 1 | 梯度累积步数 (等效 batch_size = batch_size × grad_accum) |
| `--learning_rate` | 3e-4 | 初始学习率 (含 5 epoch 线性 warmup + cosine decay) |
| `--num_queries` | 17 | 每尺度 query 数 (总 51 = 17×3) |
| `--use_amp` | False | 混合精度训练 |
| `--resume` | None | 恢复训练的 checkpoint 路径 |

**Loss 权重** (V4):
- bbox: 2.0, giou: 1.5, target: 3.0, state: 2.0, aux: 0.4
- 匹配策略: One-to-Many Top-K (K=3)
- state loss: Pair-weighted CE + Dice (稀有组合如 Aircraft/Vessel+Replaced 权重 3x)

### 评估

```bash
# 评估实例级模型
python HICD/changedetection/script/eval_unified.py \
    --mode instance \
    --checkpoint HICD/outputs/experiment_1/best.pth

# 评估像素级 BCD 方法 (二值预测图)
python HICD/changedetection/script/eval_unified.py \
    --mode pixel --pred_dir /path/to/binary_predictions

# 评估 SCD 方法 (语义变化图, 像素值 = train_id)
python HICD/changedetection/script/eval_unified.py \
    --mode scd --pred_dir /path/to/scd_predictions
```

### Docker 部署

```bash
# 构建镜像
docker build -t mambacd:latest .

# 导出 (传到服务器)
docker save mambacd:latest | gzip > mambacd.tar.gz

# 服务器运行
docker run --gpus all \
    -v /data/0617final:/workspace/HICD/0617final \
    -v /data/weights:/workspace/HICD/weights \
    -v /data/outputs:/workspace/HICD/outputs \
    mambacd:latest
```

## ICD 统一评估协议

**问题**: 实例级方法输出检测框 + 类别, 像素级方法输出逐像素分割图, 无法直接对比。

**解决**: ICD (Instance-level Change Detection) 协议 — 以目标为中心评估 "是否检测到该目标变化"。

### 核心思路

1. 对每个 GT 实例, 计算预测变化区域与 GT 框的 IoU
2. IoU ≥ 阈值 → 检测成功
3. 支持多个 IoU 阈值 (0.1~0.9), 计算 ICD-mAP

### 大目标特殊处理

遥感影像中跑道等大范围目标可能只有局部受损。面积 > 5000 像素的实例自动沿长轴切分为 4 个子区域, 每个子区域独立评估, 避免大面积目标因局部命中率低而被误判为漏检。

### 三种评估模式

| 模式 | 输入格式 | 评估逻辑 |
|------|----------|----------|
| `instance` | 模型 checkpoint | 匈牙利匹配预测框与 GT 框 |
| `pixel` | 二值预测图 (0/1) | GT 框内变化像素 IoU |
| `scd` | 语义图 (train_id) | 按 classes.csv 精确匹配 (target, state) |

### 公平性保障

- 实例级和像素级方法使用同一套 ICD-Precision/Recall/F1/mAP
- 大目标分段评估, 避免对像素级方法不公平
- 未变化实例上的误报单独计入 FP
- SCD 模式按 train_id 精确匹配: GT building_damaged 只接受预测图中值为 29 的像素

## 模型 I/O

### 输入

```python
pre_data:  Tensor (B, 3, 512, 512)  # 前时相 RGB, 归一化到 [0,1]
post_data: Tensor (B, 3, 512, 512)  # 后时相 RGB
```

### 输出

```python
outputs = model(pre_data, post_data)
# pred_boxes:   (B, Q, 4)    归一化 [cx, cy, w, h]
# pred_target:  (B, Q, 16)   目标类型 logits
# pred_state:   (B, Q, 6)    变化状态 logits
# query_feats:  (B, Q, 128)  实例特征
# aux_outputs:  list[dict]    辅助层预测
```

### 推理 (含弹坑感知后处理)

推理后自动执行: 弹坑中心点落在 Runway/Taxiway/Apron bbox 内 → 状态升级为 Damaged。

```python
results = model.inference(pre_data, post_data, confidence_threshold=0.3)
# results[b]['boxes']:   (K, 4)  过滤后 bbox
# results[b]['targets']: (K,)    目标类型索引
# results[b]['states']:  (K,)    变化状态索引
# results[b]['scores']:  (K,)    置信度
```

## 类别映射表

| target_idx | Target | train_id 范围 | 状态 |
|-----------|--------|--------------|------|
| 0 | Farmland | 1-2 | NoChange, Damaged |
| 1 | Runway | 3-7 | NoChange, Damaged, Reduced, Added, Extended |
| 2 | Taxiway | 8-12 | 同上 |
| 3 | Apron | 13-17 | 同上 |
| 4 | Highway | 23-27 | 同上 |
| 5 | Building | 28-32 | 同上 |
| 6 | Tank | 53-56 | NoChange, Damaged, Reduced, Added |
| 7 | Aircraft | 57-61 | NoChange, Damaged, Reduced, Added, Replaced |
| 8 | Vessel | 62-66 | 同上 |
| 9 | Crater | 67 | 无状态 |

> 已移除 6 个稀有类别 (Bridge/Shelter/Tower/Pier/Dock/VehicleRevet, 共 620 实例), 详见 `instances.json.bak`。


## V5 更新 (2026-08-03) — 双分支变化检测

### 核心改动

V4 是纯实例级检测，大范围目标（跑道、停机坪）的局部损坏无法精确定位。V5 新增语义分割分支，与实例检测分支并行，由 DatasetConfig 决定每个类别走哪条分支。

| 改动 | 说明 |
|------|------|
| Task-Specific Adapters | 两个轻量适配层（LayerNorm + 1×1 Conv），让实例/语义分支看到不同的特征视图，避免梯度冲突 |
| Semantic Segmentation Head | 轻量 FPN + 双头（目标区域图 + 变化状态图），输出 H/4 分辨率像素级分割 |
| DatasetConfig Branch Routing | YAML 指定每个类别走哪个分支（Building→instance，Runway→semantic） |
| Dual-Branch Joint Loss | 实例损失 + CE + Dice，加权求和 |
| Dataset V5 | 同时加载 bbox（instances.json）和像素级标注（label TIF），语义分支直接读现有像素标签 |

### 分支路由（0617final 默认）

| 分支 | 负责类别 | 输出格式 |
|------|---------|---------|
| 实例检测 | Building, Aircraft, Tank, Vessel, Crater | bbox + target + state |
| 语义分割 | Runway, Taxiway, Apron, Highway, Farmland | 像素级 target_map + state_map |

### V5 新增文件

```
HICD_v5/changedetection/
├── models/
│   ├── HICD_v5.py                  # 主模型（双分支架构）
│   ├── TaskAdapter.py              # 任务特定适配器
│   ├── SemanticSegmentationHead.py # 语义分割头（FPN + 双头）
│   └── DualBranchLoss.py           # 双分支联合损失
├── datasets/
│   └── dataset_v5.py               # V5 数据集（bbox + 像素标注）
└── script/
    └── train_full_v5.py            # V5 训练脚本
```

### V5 训练

```bash
cd /mnt/f/mambacd/home
export PYTHONPATH="/mnt/f/mambacd/home:$PYTHONPATH"

python HICD_v5/changedetection/script/train_full_v5.py \
    --dataset 0617final \
    --data_dir HICD/0617final \
    --batch_size 4 --grad_accum 4 \
    --learning_rate 3e-4 --max_epochs 100 \
    --w_instance 1.0 --w_semantic 1.0 \
    --clip_mode both --clip_unfreeze_epoch 20 \
    --use_amp --exp_name v5_dual_branch
```

### V5 参数量

| 组件 | 参数量 | 说明 |
|------|--------|------|
| Siamese VSSM Backbone | ~30M | 共享，不变 |
| ChangeDecoder | ~10M | 不变 |
| CLIP Text Encoder | ~63M | 冻结 |
| Task Adapters | ~0.1M | 新增，极轻量 |
| Instance Head | ~5M | 沿用 V4 |
| Semantic Head | ~3M | 新增，轻量 FPN + 双头 |
| **总计** | ~109M | 比 V4 多 ~3M |

## 版本历史

| 版本 | 日期 | 主要改动 |
|------|------|----------|
| V1 | 2026-07-28 | 初始架构: VSSM-tiny + CLIP + Hungarian 匹配 |
| V2 | 2026-07-29 | 多尺度 FPN (p1/p2/p3), PositionalEncoding2D, 数据增强 |
| V3 | 2026-07-30 | One-to-Many Top-K 匹配, loss 重平衡, CLIP 解冻, VSSM-small, 精简至 10 类, 变化注意力, 弹坑感知传播 |
| V4 | 2026-07-30 | SD-SSM 差值分支, Context-SSM 局部上下文注入, Pair-weighted state loss, bbox 转换 bug 修复 |
| V4.2 | 2026-07-31 | SparseChangeGate: SD-SSM 分支稀疏软阈值门控 (受 SpikeAdapter/CVPR2026 SNN 启发) |
| V5 | 2026-08-03 | 双分支解码器：实例检测 + 语义分割，Task-Specific Adapters，DatasetConfig 分支路由，Dual-Branch Loss |

## 致谢

- [VMamba](https://github.com/MzeroMiko/VMamba) — VSSM 骨干网络
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — 文本编码器



