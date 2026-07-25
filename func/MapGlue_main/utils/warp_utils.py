from typing import Dict
import numpy as np
import cv2
from .io_utils import write_geotiff

# def warp_nonrigid_to_reference(img2_info, tform, img1_info, out_path):
#     import rasterio
#     from skimage.transform import warp

#     H1, W1 = img1_info["height"], img1_info["width"]
#     C2 = img2_info["data"].shape[0]

#     registered = np.zeros((C2, H1, W1), dtype=img2_info["data"].dtype)
#     for b in range(C2):
#         registered[b] = warp(img2_info["data"][b], tform, output_shape=(H1, W1))

#     prof = img1_info["profile"].copy()
#     prof.update({
#         "transform": img1_info["transform"],
#         "crs": img1_info["crs"],
#         "compress": "lzw",
#         "tiled": True,
#         "blockxsize": 256,
#         "blockysize": 256
#     })

#     with rasterio.open(out_path, "w", **prof) as dst:
#         dst.write(registered)

#     return out_path

'''
def warp_affine_to_reference(img2_info: Dict, M: np.ndarray, img1_info: Dict, out_path: str):
    """
    仅输出img2变换后的最小外接矩形，彻底去掉黑色背景区域。
    """
    H2, W2 = img2_info["height"], img2_info["width"]
    C2 = img2_info["data"].shape[0]
    orig_dtype = img2_info["data"].dtype
    
    # --- 1. 计算img2变换后的边界（带缓冲）---
    corners = np.array([
        [0, 0, 1], 
        [W2, 0, 1], 
        [W2, H2, 1], 
        [0, H2, 1]
    ], dtype=np.float64)  # 使用float64提高精度
    
    transformed = (M @ corners.T).T
    
    # 计算边界（带2像素缓冲）
    padding = 2
    min_x = int(np.floor(transformed[:, 0].min())) - padding
    max_x = int(np.ceil(transformed[:, 0].max())) + padding
    min_y = int(np.floor(transformed[:, 1].min())) - padding
    max_y = int(np.ceil(transformed[:, 1].max())) + padding
    
    # 确保尺寸有效
    new_width = max(1, max_x - min_x)
    new_height = max(1, max_y - min_y)
    
    # --- 2. 调整变换矩阵 ---
    # 将2x3矩阵转为3x3齐次坐标矩阵
    M_homo = np.eye(3)
    M_homo[:2, :] = M
    
    # 创建平移矩阵来调整输出位置
    adjust_matrix = np.eye(3)
    adjust_matrix[0, 2] = -min_x
    adjust_matrix[1, 2] = -min_y
    
    # 组合变换矩阵
    M_adj = adjust_matrix @ M_homo
    
    # --- 3. 执行变换 ---
    registered = np.zeros((C2, new_height, new_width), dtype=orig_dtype)
    
    for b in range(C2):
        interp = cv2.INTER_NEAREST if np.issubdtype(orig_dtype, np.integer) else cv2.INTER_LINEAR
        
        # 使用warpPerspective处理3x3矩阵
        warped = cv2.warpPerspective(
            img2_info["data"][b], 
            M_adj, 
            (new_width, new_height),
            flags=interp, 
            borderValue=0
        )
        registered[b] = warped.astype(orig_dtype)
    
    # --- 4. 更新地理变换 ---
    new_transform = img1_info["transform"] * Affine.translation(min_x, min_y)
    
    # --- 5. 更新profile并写出 ---
    prof = img1_info["profile"].copy()
    prof.update({
        "height": new_height,
        "width": new_width,
        "transform": new_transform,
        "dtype": orig_dtype,
    })
    
    # 保持原有的nodata设置
    if "nodata" in img2_info.get("profile", {}):
        prof["nodata"] = img2_info["profile"]["nodata"]
    
    # 添加压缩选项（可选）
    prof.update({
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256
    })
    
    write_geotiff(out_path, registered, prof)
    return out_path
'''

# def warp_affine_to_reference(img2_info: Dict, M: np.ndarray, img1_info: Dict, out_path: str):
#     """将 img2 通过仿射矩阵 M 变换到 img1 的像素网格/空间范围，并写出 GeoTIFF。
#     - 输出影像的 CRS/transform 直接复用 img1 的 profile（与参考影像严格对齐）。
#     - 逐波段使用 cv2.warpAffine 插值到 (W1, H1)。
#     """
#     H1, W1 = img1_info["height"], img1_info["width"]
#     C2 = img2_info["data"].shape[0]

    
#     # registered = np.zeros((C2, H1, W1), dtype=img2_info["data"].dtype)

#     # 准备输出数组（使用与原图相同 dtype）
#     # orig_dtype = img2_info.get("dtype", img2_info["data"].dtype)
#     orig_dtype = img2_info["data"].dtype
#     print(f"Original dtype: {orig_dtype}")  # 调试信息

#     registered = np.zeros((C2, H1, W1), dtype=orig_dtype)

#     # for b in range(C2):
#     #     registered[b] = cv2.warpAffine(
#     #         img2_info["data"][b], M, (W1, H1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
#     #     )
#     for b in range(C2):
#         # 如果原始是整数类型，考虑使用最近邻插值
#         if np.issubdtype(orig_dtype, np.integer):
#             interp_flag = cv2.INTER_NEAREST
#         else:
#             interp_flag = cv2.INTER_LINEAR
            
#         warped = cv2.warpAffine(
#             img2_info["data"][b], M, (W1, H1), 
#             flags=interp_flag, 
#             borderMode=cv2.BORDER_CONSTANT, 
#             borderValue=0
#         )
        
#         # 确保数据类型一致
#         registered[b] = warped.astype(orig_dtype)



#     prof = img1_info["profile"].copy()
#     prof.update({
#         "transform": img1_info["transform"],
#         "crs": img1_info["crs"],
#         "dtype": orig_dtype,       # 明确指定数据类型
#         "compress": "lzw",         # 开启压缩，体积大幅缩小
#         "tiled": True,             # 瓦片存储，加快读取
#         "blockxsize": 256,         # 合理的瓦片大小
#         "blockysize": 256
#     })
#     write_geotiff(out_path, registered, prof)
#     return out_path

def warp_affine_to_reference(src_img, M, ref_img, output_path):
    """
    将 src_img 根据仿射矩阵 M 变换到 ref_img 的坐标系下，并保存。
    """
    import numpy as np
    import cv2
    import rasterio

    # 1. 获取参考影像的尺寸 (H, W)
    # 既然传进来的是字典，直接从字典取
    h_ref = ref_img['height']
    w_ref = ref_img['width']
    
    # 2. 获取源影像数据 (修复核心)
    # src_img 是一个字典，数据通常存在 'image' 键里
    if isinstance(src_img, dict):
        if 'image' in src_img:
            src_data = src_img['image']
        elif 'data' in src_img:
            src_data = src_img['data'] # 防止有人命名为 data
        else:
            raise ValueError(f"src_img 是字典，但找不到影像数据键 (image/data)。现有键: {src_img.keys()}")
    else:
        # 如果万一传进来的是 rasterio 对象
        src_data = src_img.read()
    
    # 3. 调整维度顺序给 OpenCV 使用: (H, W, Bands)
    # Rasterio 读取的数据通常是 (Bands, H, W)
    if src_data.ndim == 3:
        # 变成 (H, W, Bands)
        src_data_cv = src_data.transpose(1, 2, 0)
    else:
        # 单波段 (H, W)
        src_data_cv = src_data
    
    # ================= 类型检查与转换 (uint32 -> float32) =================
    # print(f"DEBUG: 源数据类型: {src_data_cv.dtype}, 形状: {src_data_cv.shape}")
    
    if src_data_cv.dtype == np.uint32 or src_data_cv.dtype == np.int32:
        # print("警告: 转换 uint32/int32 -> float32 以适配 OpenCV")
        src_data_cv = src_data_cv.astype(np.float32)
    # ======================================================================

    # 4. 执行仿射变换
    warped_cv = cv2.warpAffine(
        src_data_cv, 
        M, 
        (w_ref, h_ref), 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=0
    )

    # 5. 转回 Rasterio 格式: (Bands, H, W)
    if warped_cv.ndim == 3:
        warped_data = warped_cv.transpose(2, 0, 1)
    else:
        # OpenCV 输出单波段时可能丢掉维度，补回来 (1, H, W)
        warped_data = warped_cv[np.newaxis, :, :]

    # 6. 更新 Profile 并保存
    # 这里的 ref_img 也是字典，取 profile 属性
    if 'profile' in ref_img:
        profile = ref_img['profile'].copy()
    else:
        # 如果没有 profile，手动构建一个基础的
        profile = {
            'driver': 'GTiff',
            'dtype': warped_data.dtype,
            'count': warped_data.shape[0],
            'height': h_ref,
            'width': w_ref,
            'crs': ref_img.get('crs'),
            'transform': ref_img.get('transform')
        }

    # 强制更新关键属性
    profile.update({
        'count': warped_data.shape[0],
        'dtype': warped_data.dtype, 
        'height': h_ref,
        'width': w_ref,
        # 确保使用参考影像的地理信息
        'transform': ref_img['transform'],
        'crs': ref_img['crs']
    })

    # print(f"✅ 正在写入配准结果: {output_path}")
    # with rasterio.open(output_path, 'w', **profile) as dst:
    #     dst.write(warped_data)
        
    # return output_path
    return {
    "image": warped_data,
    "transform": ref_img['transform'],
    "crs": ref_img['crs'],
    "height": h_ref,
    "width": w_ref
    }






import os
from osgeo import gdal

def build_gcps_and_warp(src_path, dst_path, tiepoints, dst_srs="EPSG:4326"):
    """
    使用 GDAL TPS 变换进行图像校正
    :param src_path: 输入影像路径
    :param dst_path: 输出影像路径
    :param tiepoints: [(x_src, y_src, x_dst, y_dst), ...] 格式的同名点
    :param dst_srs: 目标投影坐标系（默认 WGS84）
    """
    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise IOError(f"无法打开输入影像 {src_path}")

    # 构造 GCP 列表
    gcps = []
    for (x_src, y_src, x_dst, y_dst) in tiepoints:
        gcp = gdal.GCP()
        gcp.GCPX = x_dst
        gcp.GCPY = y_dst
        gcp.GCPZ = 0
        gcp.GCPPixel = x_src
        gcp.GCPLine = y_src
        gcps.append(gcp)

    # 创建带 GCP 的临时 VRT
    tmp_vrt = dst_path.replace(".tif", "_gcps.vrt")
    gdal.Translate(
        tmp_vrt,
        src_ds,
        GCPs=gcps,
        outputSRS=dst_srs
    )

    # 使用 TPS（Thin Plate Spline）进行重采样
    gdal.Warp(
        dst_path,
        tmp_vrt,
        tps=True,  # 启用 Thin Plate Spline
        dstSRS=dst_srs,
        resampleAlg="cubic",  # 可换为 bilinear, lanczos 等
        multithread=True,
        warpMemoryLimit=512
    )

    os.remove(tmp_vrt)
    print(f"影像已完成 TPS 校正: {dst_path}")
