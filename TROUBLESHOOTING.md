# MambaCD 环境配置与踩坑记录

> 日期: 2026-07-28
> 环境: Windows 10 + WSL2 (Ubuntu 26.04) + NVIDIA TITAN RTX

## 1. WSL 安装与网络配置

### 1.1 WSL 安装
- Windows 10 内置 WSL 版本过旧，不支持 `--import`、`--export` 等命令
- 需要从 Microsoft Store 安装新版 WSL：`wsl --install --web-download --no-distribution`
- 安装后需要**重启系统**才能生效

### 1.2 DNS 问题（VPN 干扰）
**现象**: WSL 内无法访问 `archive.ubuntu.com`，DNS 解析到 `198.18.0.106`（VPN 本地 IP）

**原因**: 宿主机有 VPN 适配器，WSL 的 DNS 被劫持

**解决**:
```bash
# 修改 /etc/resolv.conf
nameserver 10.111.147.112  # 宿主机实际 DNS
nameserver 8.8.8.8

# 防止 WSL 重启覆盖
# /etc/wsl.conf
[network]
generateResolvConf = false
```

---

## 2. Python 环境

### 2.1 Python 版本兼容性
**问题**: Ubuntu 26.04 默认 Python 3.14，但 numpy==1.23.0、triton==2.1.0 等包不兼容

**解决**: 使用 Miniconda 安装 Python 3.11
```bash
# 需要先接受 Anaconda TOS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n mamba python=3.11 -y
```

### 2.2 编译依赖
**问题**: contourpy、GDAL 等包需要 C++ 编译器

**解决**:
```bash
sudo apt-get install -y build-essential python3.14-dev pkg-config libgdal-dev
pip install GDAL==$(gdal-config --version)  # 版本要匹配系统 libgdal
```

---

## 3. PyTorch 与 CUDA

### 3.1 CUDA 版本不匹配
**现象**: `CUDA initialization: The NVIDIA driver on your system is too old`

**原因**: pip 默认安装了 CUDA 13.0 编译的 PyTorch，但驱动只支持 CUDA 12.6

**解决**:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3.2 CUDA 扩展编译
**问题**: `selective_scan_cuda_oflex` 未定义

**解决**:
```bash
cd kernels/selective_scan
pip install -e . --no-build-isolation  # 需要 --no-build-isolation 否则找不到 torch
```

---

## 4. 数据维度问题（核心坑点）

### 4.1 GDAL 读取格式
**问题**: GDAL `ReadAsArray()` 返回 `(C, H, W)` 格式，但代码假设 `(H, W, C)`

**解决**: `_read_tif` 统一返回 `(H, W, C)`
```python
def _read_tif(self, path):
    arr = ds.ReadAsArray()
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)  # (H,W,3)
    elif arr.ndim == 3:
        if arr.shape[0] > 3:
            arr = arr[:3]  # 取前3个波段
        arr = np.transpose(arr, (1, 2, 0))  # (C,H,W)->(H,W,C)
    return arr
```

### 4.2 数据增强 axes 错误
**问题**: `_random_augment` 的 `np.rot90(axes=(1,2))` 是按 `(C,H,W)` 设计的

**解决**: 改为 `axes=(0,1)` 适配 `(H,W,C)` 格式
```python
pre_img = np.rot90(pre_img, k, axes=(0, 1)).copy()  # H-W 平面旋转
```

### 4.3 CLIP 文本特征维度不匹配
**现象**: `mat1 and mat2 shapes cannot be multiplied (204x128 and 512x16)`

**原因**: CLIPTextEncoder 输出 512 维，但视觉特征是 128 维

**解决**: 修改 `embed_dim=128`（原来 512）
```python
# HierarchicalSCD_Instance.py
self.clip_text_encoder = CLIPTextEncoder(
    clip_model=clip_model, embed_dim=128,  # 原来是 512
    freeze=True, pretrained_path=clip_weights_path
)

# CrossAttentionFusion.py
self.cross_attn = TextVisualCrossAttention(
    visual_dim=128, text_dim=128,  # 原来是 512
    num_heads=8, dropout=0.1
)
```

---

## 5. 模型初始化配置

### 5.1 cfg.defaulter() 不存在
**问题**: yacs 配置对象没有 `defaulter()` 方法

**解决**: 手动构建 cfg_dict
```python
cfg.defrost()
vssm = cfg.MODEL.VSSM
cfg_dict = {
    'norm_layer': vssm.NORM_LAYER,
    'ssm_act_layer': vssm.SSM_ACT_LAYER,
    'mlp_act_layer': vssm.MLP_ACT_LAYER,
    'ssm_d_state': vssm.SSM_D_STATE,
    'ssm_ratio': vssm.SSM_RATIO,
    # ... 完整映射见 train_full.py
}
```

### 5.2 CLIP 权重加载格式
**问题**: 下载的权重没有 `state_dict` 键

**解决**:
```python
state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
model.load_state_dict(state_dict)
```

---

## 6. 数据集结构

### 6.1 目录结构不一致
**问题**: 原始数据是扁平结构 `image/xxx_pre_war.tif`，代码期望 `image/pre/xxx.tif`

**解决**: `train_full.py` 自动检测目录结构
```python
self.flat_structure = not os.path.isdir(os.path.join(train_dir, "pre"))

if self.flat_structure:
    pre_path = os.path.join(..., "image", img_stem + "_pre_war.tif")
else:
    pre_path = os.path.join(..., "image", "pre", stem + ".tif")
```

### 6.2 instances.json key 格式
**问题**: key 是 `train/xxx_target.tif`，图片文件名是 `xxx_pre_war.tif`

**解决**: 提取 stem 时去掉 `_target`
```python
img_stem = stem.replace('_target', '')
```

---

## 7. WSL 路径转换

**问题**: Windows 路径 `D:/CD/0617final` 在 WSL 中无法访问

**解决**: 自动转换
```python
def win_to_wsl(path):
    if path and len(path) >= 2 and path[1] == ':':
        drive = path[0].lower()
        rest = path[2:].replace('\\', '/')
        return f'/mnt/{drive}{rest}'
    return path
```

---

## 8. 依赖清单

```bash
# 系统依赖
sudo apt-get install -y build-essential python3.11-dev pkg-config libgdal-dev

# Python 依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install fvcore scipy open_clip_torch huggingface_hub
pip install GDAL==$(gdal-config --version)
pip install ninja && cd kernels/selective_scan && pip install -e . --no-build-isolation
```

---

## 9. 预训练权重

| 权重 | 用途 | 下载地址 |
|------|------|---------|
| vssm1_tiny_224.pth | VSSM 骨干网络 | [VMamba GitHub](https://github.com/MzeroMako/VMamba) |
| open_clip_pytorch_model.bin | CLIP 文本编码器 | `huggingface_hub.hf_hub_download('timm/vit_base_patch16_clip_224.openai', ...)` |

---

## 10. 训练命令

```bash
cd /mnt/f/mambacd/home
export PYTHONPATH="/mnt/f/mambacd/home:$PYTHONPATH"
source ~/miniconda/bin/activate && conda activate mamba

python MambaCD/changedetection/script/train_full.py \
    --data_dir "D:/CD/0617final" \
    --scenes "Airports,Ports,Urban-Rural Areas" \
    --clip_weights_path MambaCD/weights/open_clip_pytorch_model.bin \
    --cfg MambaCD/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml \
    --batch_size 4 \
    --max_epochs 100 \
    --learning_rate 1e-4 \
    --use_amp \
    --output_dir MambaCD/outputs \
    --exp_name full_train_v1
```
