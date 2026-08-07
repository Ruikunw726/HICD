# HICD v6 — 层级实体变化检测

基于 LWGANet 轻量级骨干网络的双分支遥感变化检测框架。

## 核心改进

1. **LWGANet 骨干网络** — 替代 VMamba，轻量级分组注意力网络（15.7M 参数），多尺度特征提取能力更强
2. **邻域特征融合 (NFA)** — 为每个时相注入多尺度空间上下文，弥补骨干局部信息不足
3. **时序融合模块 (Temporal Fusion)** — 对齐并融合双时相多尺度特征
4. **任务适配器 (Task Adapters)** — 分离实体/语义分支的特征路径，解决梯度冲突问题
5. **DETR 实体检测头** — 基于 Transformer 解码器的实体级检测，输出边界框 + 目标类别 + 损坏状态
6. **双头语义分割** — FPN 基础上的双头结构，分别预测目标类型和损坏状态（数据集有像素标注时启用）
7. **模块化重构** — backbone/modules/heads/losses 清晰分层，方便替换和消融
8. **双分支联合损失** — 实体分支 Hungarian 匹配损失（class + bbox + giou + state）+ 语义分支 Masked CE/Dice
9. **数据集配置化** — 每个数据集一个 YAML，模型按 --dataset 自加载
10. **空分支保护** — 语义分支无标注时自动跳过，loss 函数加 NaN 保护

## 数据集

**0617final** — 9,687 对 512x512 VHR 双时相影像

| 场景 | Train | Val | Test | 实体数 |
|------|-------|-----|------|--------|
| Airports | 660 | 220 | 221 | 5,958 |
| Ports | 358 | 119 | 119 | 4,868 |
| Urban-Rural Areas | 4,937 | 1,234 | 1,819 | 204,860 |

**xBD** — 灾害损坏评估数据集 (Building x 5 states)

| Split | 样本数 |
|-------|--------|
| Train | 7,253 |
| Val | 2,202 |
| Test | 1,579 |

**SECOND** — 语义变化检测数据集 (6 类目标 x 4 种状态)

## 项目结构

`
HICD_v6/
├── models/
│   ├── HICD_v6.py                  # 主模型
│   ├── backbone/
│   │   ├── lwganet.py              # LWGANet 骨干
│   │   └── norm_patch.py           # 归一化层
│   ├── modules/
│   │   ├── nfa.py                  # 邻域特征融合
│   │   ├── tfm.py                  # 时序融合模块
│   │   ├── task_adapter.py         # 任务适配器
│   │   └── clip_module.py          # 轻量 CLIP 编码器
│   ├── heads/
│   │   ├── instance_head.py        # DETR 实体检测头 (bbox + class + state)
│   │   └── semantic_head.py        # 语义分割头
│   └── losses/
│       ├── detr_loss.py            # DETR 匹配损失 (class + bbox + giou + state)
│       └── dual_branch_loss.py     # 双分支联合损失
├── datasets/
│   ├── dataset_v6.py               # 数据集与数据加载器（含图片缓存）
│   ├── stitcher.py                 # 推理时 Patch 拼接
│   └── configs/
│       ├── xbd.yaml                # xBD: 1 类目标, 5 种状态
│       ├── 0617final.yaml          # 0617final: 10 类目标, 6 种状态
│       └── second.yaml             # SECOND: 6 类目标, 4 种状态
├── evaluation/
│   └── icd_eval_v6.py              # ICD 评估协议
├── script/
│   └── train_full_v6.py            # 训练入口（含 val 评估、CSV 日志、tqdm 进度条）
└── weights/                        # 预训练权重（不入库）
`

## 快速开始

### 环境配置

`ash
conda create -n hicd python=3.11 -y && conda activate hicd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm antialiased-cnns open-clip-torch scipy
`

### 训练

`ash
cd HICD_v6
export PYTHONPATH=

python script/train_full_v6.py \
    --dataset xbd \
    --data_dir /path/to/xbd \
    --backbone L1 \
    --pretrained_weight_path weights/lwganet_l1_e299.pth \
    --batch_size 16 \
    --grad_accum 2 \
    --max_epochs 200 \
    --learning_rate 1e-4 \
    --use_amp \
    --patch_size 256 \
    --num_workers 8 \
    --exp_name v6_xbd_256
`

### 主要参数

| 参数 | 默认值 | 说明 |
|------|------|------|
| --backbone | L1 | LWGANet 规格: L0 / L1 / L2 |
| --batch_size | 4 | 批次大小 |
| --grad_accum | 4 | 梯度累积步数 |
| --max_epochs | 200 | 训练轮数 |
| --learning_rate | 1e-4 | 学习率 |
| --freeze_backbone_epochs | 10 | 前 N 轮冻结骨干 |
| --unfreeze_ratio | 0.5 | 解冻骨干层数比例 |
| --patch_size | 256 | 输入图块尺寸 |
| --use_amp | 关闭 | 混合精度训练 |
| --lambda_inst | 1.0 | 实体损失权重 |
| --lambda_sem | 1.0 | 语义损失权重 |

## 模型 I/O

### 输入

`python
img_t1: Tensor (B, 3, H, W)  # 时相 T1 RGB
img_t2: Tensor (B, 3, H, W)  # 时相 T2 RGB
`

### 输出

`python
predictions = model(img_t1, img_t2)
# instance_logits:       (B, Q, num_targets)    实体类别 logits
# instance_boxes:        (B, Q, 4)              归一化 [cx, cy, w, h]
# instance_state_logits: (B, Q, num_states)     实体状态 logits（每实例）
# target_logits:         (B, num_targets, H, W) 目标类型预测（像素级，可选）
# state_logits:          (B, num_states, H, W)  损坏状态预测（像素级，可选）
`

## 评估 (ICD 协议)

ICD 评估协议衡量三个维度:

- **实体分支 (ICD-Instance)** — bbox IoU 匹配 -> mAP, P/R/F1, Target-Acc, State-Acc
- **语义分支 (ICD-Pixel)** — 像素级评估 -> mIoU, Pixel P/R/F1, State-mIoU（需像素标注）
- **综合 (ICD-Overall)** — 两个分支的加权平均

详见 evaluation/icd_eval_v6.py

## 版本历史

| 版本 | 日期 | 主要改动 |
|------|------|----------|
| V1-V3 | 2026-07-28~30 | 见 [HICD](https://github.com/Ruikunw726/HICD) |
| V4 | 2026-07-30 | SD-SSM, Context-SSM, Pair-weighted Loss |
| V4.2 | 2026-07-31 | SparseChangeGate |
| V5 | 2026-08-03 | 双分支架构，Task Adapters，DatasetConfig YAML，Dual-Branch Loss |
| V5.1 | 2026-08-03 | Masked Dual-Branch Loss, w_instance=3.0 |
| V5.2 | 2026-08-03 | SECOND 数据集支持 |
| **V6** | **2026-08-04** | **LWGANet 骨干, NFA, Temporal Fusion, DETR 检测头, 模块化重构** |
| V6.1 | 2026-08-06 | 实例级 state 分类头、空分支 NaN 保护、图片缓存加速 |
| V6.2 | 2026-08-07 | val_loader 补全、tqdm 进度条、CSV 日志、XBD instances.json 合并 |

## 踩坑记录

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| VMamba 在变化检测上增益不明显 | Mamba 擅长长序列建模，但遥感图像空间关系更重要 | 替换为 LWGANet，分组注意力更适合空间特征提取 |
| 共享 backbone 实体/语义梯度冲突 | 实体分支噪声梯度干扰语义学习 | Task Adapters 分离特征路径 |
| DETR 训练不稳定 | Transformer 解码器收敛慢 | backbone 先冻结 10 轮，再逐步解顿 |
| mAP 始终为 0 | 评估器是空壳（return 0.0） | 重写为完整 mAP 计算 |
| 实例级检测全 0 | instances.json key 格式不匹配，boxes 未加载 | 合并 per-scene JSON，统一 key 格式 |
| 语义分支 NaN loss | XBD 无像素标注，语义 loss 在空标签上计算 | 空分支自动跳过 + NaN 保护 |
| 预计算 patch 磁盘爆炸 | .pt 存原始张量，膨胀 35 倍（15GB->536GB） | 改为在线裁剪 + 图片缓存 |
| 训练极慢（HDD） | 每个 patch 读整张图，同一图反复读 | LRU 图片缓存（每 worker 64 张） |

## 后续规划

- **架构方向**：先变化检测再目标定位（Change-first paradigm），利用双时相差异信号提升严重损毁目标的检测能力
- **共享语义层**：实例分支和像素分支共享底层语义特征，分支头各自做任务特化
- **消融实验**：可学习嵌入 vs CLIP、Task Adapter 效果、双分支 vs 单分支

## 致谢

- [LWGANet](https://github.com/LiZhengXun99/LWGANet) — 轻量级分组注意力网络
- [DETR](https://github.com/facebookresearch/detr) — 检测 Transformer
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — CLIP 文本编码器
- [VMamba](https://github.com/MzeroMiko/VMamba) — VSSM 骨干（v1-v5）
- [xBD](https://xview2.org/) — 灾害损坏评估数据集
- [SECOND](https://captain-whu.github.io/SCD/) — 语义变化检测数据集
