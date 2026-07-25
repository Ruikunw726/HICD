import numpy as np
import rasterio
from rasterio.enums import Resampling
from typing import Dict


def read_geotiff(image_path: str) -> Dict:
    """读取 GeoTIFF 及空间参考信息。"""
    with rasterio.open(image_path) as src:
        data = src.read()  # (C, H, W)
        info = {
            "data": data,
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "profile": src.profile,
            "dtype": src.dtypes[0],
            "count": src.count,
            "nodata": src.nodata,
        }
    return info

# def read_geotiff(image_path: str) -> Dict:
#     """读取 GeoTIFF 及空间参考信息（修复Docker Linux CRS兼容）。"""
#     with rasterio.open(image_path) as src:
#         data = src.read()  # (C, H, W)
#         # 核心修改：将rasterio.CRS对象转为WKT字符串（跨平台兼容）
#         crs_wkt = src.crs.to_wkt() if src.crs is not None else None
#         info = {
#             "data": data,
#             "crs": crs_wkt,  # 替换为WKT字符串，而非CRS对象
#             "transform": src.transform,
#             "bounds": src.bounds,
#             "width": src.width,
#             "height": src.height,
#             "profile": src.profile,
#             "dtype": src.dtypes[0],
#             "count": src.count,
#             "nodata": src.nodata,
#         }
#     # 验证CRS读取结果（调试用，可删除）
#     if crs_wkt is None:
#         print(f"警告：{image_path} 实际无CRS！")
#     else:
#         print(f"{image_path} CRS读取成功（WKT格式）")
#     return info


def write_geotiff(path: str, data: np.ndarray, ref_profile: Dict):
    """按参考 profile 写出 GeoTIFF。data 形状为 (C, H, W)。"""
    profile = ref_profile.copy()
    profile.update({
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "compress": profile.get("compress", "deflate"),
        "tiled": True,
        "blockxsize": min(512, data.shape[2]),
        "blockysize": min(512, data.shape[1]),
        "BIGTIFF": "IF_SAFER",
    })
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def to_uint8_for_cv(img: np.ndarray) -> np.ndarray:
    """将影像块转换为 uint8(H, W, 3) 以便 OpenCV/模型推理。
    规则：
    - 单波段：按 2/5/8/16 位动态范围线性拉伸到 0-255，并复制到 3 通道。
    - 多波段：取前 3 个通道，各自线性拉伸到 0-255。
    - 若数据全常数，直接返回零阵。
    输入 img 形状可为 (H,W) 或 (H,W,C)。
    """
    def _scale_to_uint8(ch: np.ndarray) -> np.ndarray:
        ch = ch.astype(np.float32)
        vmin, vmax = np.percentile(ch, [2, 98]) if np.any(np.isfinite(ch)) else (0, 1)
        if vmax <= vmin:
            vmax = vmin + 1.0
        ch = (ch - vmin) / (vmax - vmin)
        ch = np.clip(ch, 0, 1) * 255.0
        return ch.astype(np.uint8)

    if img.ndim == 2:
        u = _scale_to_uint8(img)
        return np.stack([u, u, u], axis=-1)
    elif img.ndim == 3:
        C = img.shape[2]
        if C >= 3:
            bands = [ _scale_to_uint8(img[..., i]) for i in range(3) ]
            return np.stack(bands, axis=-1)
        elif C == 2:
            bands = [ _scale_to_uint8(img[..., 0]), _scale_to_uint8(img[..., 1]), _scale_to_uint8(img[..., 0]) ]
            return np.stack(bands, axis=-1)
        else:
            u = _scale_to_uint8(img[..., 0])
            return np.stack([u, u, u], axis=-1)
    else:
        raise ValueError("Unsupported image shape for uint8 conversion")

