# HICD — Hierarchical Instance Change Detection

基于 Mamba (VSSM) + CLIP 文本引导的层级实例级遥感变化检测框架。

## 核心创新

1. **Mamba 骨干网络** — 用 State Space Model 替代 Transformer 做遥感变化检测, 线性复杂度处理高分辨率影像
2. **CLIP 文本引导** — 利用视觉-语言模型的语义先验, 支持零样本/可扩展类别
3. **层级检测头** — 目标类型 → 变化状态的两层分类, 通过有效性矩阵约束合法组合
4. **多尺度 FPN** — 自顶向下路径 + 侧向连接, 处理 5000 倍尺度差异
5. **辅助层损失** — 中间 decoder 层监督, 加速收敛
6. **ICD 统一评估协议** — 实例级评估框架, 公平对比像素级和实例级方法

## 数据集

**0617final** — 9,687 对 512×512 VHR 双时相影像

| 场景 | Train | Val | Test | 实例数 |
|------|-------|-----|------|--------|
| Airports | 660 | 220 | 221 | 5,958 |
| Ports | 358 | 119 | 119 | 4,868 |
| Urban-Rural Areas | 4,937 | 1,234 | 1,819 | 204,860 |

**类别体系**: 16 目标类型 × 6 变化状态, 68 个非背景 train_id。详见 `0617final/classes.csv`。

## 项目结构

```
MambaCD/
├── 0617final/                 # 数据集 (含 instances.json)
├── weights/                   # 预训练权重
│   ├── vssmtiny_dp01_ckpt_epoch_292.pth   # VSSM backbone
│   └── open_clip_pytorch_model.bin        # CLIP 文本编码器
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

### 训练

```bash
cd /path/to/MambaCD/..
export PYTHONPATH="$(pwd):$PYTHONPATH"

python MambaCD/changedetection/script/train_full.py \
    --data_dir MambaCD/0617final \
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

**Loss 权重** (当前最优):
- bbox: 3.0, giou: 2.0, target: 2.0, state: 1.5, aux: 0.4

### 评估

```bash
# 评估实例级模型
python MambaCD/changedetection/script/eval_unified.py \
    --mode instance \
    --checkpoint MambaCD/outputs/experiment_1/best.pth

# 评估像素级 BCD 方法 (二值预测图)
python MambaCD/changedetection/script/eval_unified.py \
    --mode pixel --pred_dir /path/to/binary_predictions

# 评估 SCD 方法 (语义变化图, 像素值 = train_id)
python MambaCD/changedetection/script/eval_unified.py \
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
    -v /data/0617final:/workspace/MambaCD/0617final \
    -v /data/weights:/workspace/MambaCD/weights \
    -v /data/outputs:/workspace/MambaCD/outputs \
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

### 推理

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
| 4 | Bridge | 18-22 | 同上 |
| 5 | Highway | 23-27 | 同上 |
| 6 | Building | 28-32 | 同上 |
| 7 | Shelter | 33-37 | 同上 |
| 8 | Tower | 38-42 | 同上 |
| 9 | Pier | 43-47 | 同上 |
| 10 | Dock | 48-52 | 同上 |
| 11 | Tank | 53-56 | NoChange, Damaged, Reduced, Added |
| 12 | Aircraft | 57-61 | NoChange, Damaged, Reduced, Added, Replaced |
| 13 | Vessel | 62-66 | 同上 |
| 14 | Crater | 67 | 无状态 |
| 15 | VehicleRevet | 68 | 无状态 |

## 致谢

- [VMamba](https://github.com/MzeroMiko/VMamba) — VSSM 骨干网络
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — 文本编码器