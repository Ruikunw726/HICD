import os
import json
from typing import Tuple
from shapely.geometry import Polygon, Point
import numpy as np
import cv2

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# from pyproj import CRS, Transformer

from rasterio.transform import xy

# from .utils.gdal_proj import CRS, Transformer
from utils.io_utils import read_geotiff
from utils.geo_utils import common_area_wgs84, generate_lonlat_grid, wgs84_box_to_crs
from utils.crop_utils import bbox_to_window, crop_block
from utils.registration import FastMapGlueMatcher, collect_control_points, estimate_affine_global
from utils.warp_utils import warp_affine_to_reference
# from .utils.chessboard import generate_chessboard_with_fix
from utils.geojson import save_points_as_geojson

my_base_path = "D:\MapGlue-main"


# def make_transformer(src_crs, dst_crs):
#     return Transformer.from_crs(CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs), always_xy=True)


def poly_projector(dst_crs):
    """返回一个函数：将 WGS84 多边形投影到 dst_crs。"""
    from utils.geo_utils import wgs84_box_to_crs
    def _fn(poly: Polygon):
        return wgs84_box_to_crs(poly, dst_crs)
    return _fn


def register_images(
    img1_path: str | dict,
    img2_path: str | dict,
    output_path: str,
    grid_step_deg: float = 0.01,
    model_path: str = './weights/fastmapglue_model.pt',
    num_keypoints: int = 1024,
    save_control_points: bool = True,
    generate_chessboard: bool = True,
    chessboard_path: str = None,
):
    print("registration started")
    
    if isinstance(img1_path, dict):
        img1 = img1_path
    else:
        img1 = read_geotiff(img1_path)

    if isinstance(img2_path, dict):
        img2 = img2_path
    else:
        img2 = read_geotiff(img2_path)

    from rasterio.crs import CRS
    from osgeo import gdal
    def fix(img_dict,file_path):
        if img_dict.get('crs') is None:
            try:
                ds = gdal.Open(file_path)
                if ds:
                    wkt = ds.GetProjection()
                    if wkt:
                        img_dict['crs']=CRS.from_wkt(wkt)
            except Exception as e:
                print(f"reading faild:{e}")
    fix(img1,img1_path)
    fix(img2,img2_path)

    # ===================== 适配 crop 输出：CRS 字符串转对象 =====================
    for img in [img1, img2]:
        if isinstance(img['crs'], str):
            img['crs'] = CRS.from_wkt(img['crs'])

    # ===================== 适配 crop 输出：transform 正确转为 Affine =====================
    from rasterio.transform import Affine
    for img in [img1, img2]:
        if isinstance(img['transform'], (list, tuple)):
            gt = img['transform']
            # 正确顺序：GDAL 6参数 → Affine
            img['transform'] = Affine(gt[1], gt[2], gt[0],   # a, b, c
                                      gt[4], gt[5], gt[3])   # d, e, f

    # ===================== 适配 crop 输出：自动计算 bounds =====================
    for img in [img1, img2]:
        if 'bounds' not in img:
            transform = img['transform']
            
            # 判断是 Affine 对象还是 GDAL 元组
            if hasattr(transform, 'a'):  # Affine 对象
                # Affine 属性: a, b, c, d, e, f = (resx, xrot, left, yrot, resy, top)
                left   = transform.c      # 左上角 x
                top    = transform.f      # 左上角 y（最大 y）
                resx   = transform.a      # x 分辨率
                resy   = transform.e      # y 分辨率（注意是 e，不是 f！）
            else:  # GDAL 元组 (c, a, b, f, d, e)
                left   = transform[0]
                resx   = transform[1]
                top    = transform[3]
                resy   = transform[5]
            
            right  = left + img['width'] * resx
            bottom = top + img['height'] * resy  # 当 resy 为负时，bottom < top
            
            img['bounds'] = (left, bottom, right, top)
            
            # 验证
            # print(f"计算 bounds: ({left}, {bottom}, {right}, {top})")
            # print(f"  width: {right-left}, height: {top-bottom}")
            
    # print("参考影像:", img1_path)
    # print("待配准影像:", img2_path)
    # print(f"CRS1={img1['crs']}, CRS2={img2['crs']}")
    # for i, img in enumerate([img1, img2], 1):
    #     print(f"img{i}:")
    #     print(f"  crs: {img['crs']}")
    #     print(f"  crs类型: {type(img['crs'])}")
    #     print(f"  bounds: {img['bounds']}")
    #     print(f"  transform: {img['transform']}")
        
    #     # 测试坐标转换
    #     test_x = img['bounds'][0]  # left
    #     test_y = img['bounds'][1]  # bottom
    #     print(f"  测试投影坐标 ({test_x}, {test_y}) -> WGS84:")
        
    #     try:
    #         # 反向转换（投影 -> WGS84）
    #         from pyproj import Transformer
    #         transformer = Transformer.from_crs(img['crs'], "EPSG:4326", always_xy=True)
    #         lon, lat = transformer.transform(test_x, test_y)
    #         print(f"    -> ({lon}, {lat})")
    #     except Exception as e:
    #         print(f"    转换失败: {e}")
    # print("\n=== 步骤2：计算公共区域（WGS84） ===")
    inter_wgs84 = common_area_wgs84(img1, img2)
    
    from shapely.geometry import box
    if hasattr(inter_wgs84, 'bounds'):
        minx, miny, maxx, maxy = inter_wgs84.bounds
        inter_wgs84 = box(minx, miny, maxx, maxy)
    # from shapely.geometry import box
    # inter_wgs84 = box(*img1['bounds'])
    if inter_wgs84 is None or inter_wgs84.is_empty:
        raise RuntimeError("两幅影像无重叠区域，无法配准。")
    # print("公共区域 bbox:", inter_wgs84.bounds)
    # print(f"inter_wgs84 类型: {type(inter_wgs84)}")
    # print(f"inter_wgs84: {inter_wgs84}")
    # print(f"inter_wgs84.bounds: {inter_wgs84.bounds if hasattr(inter_wgs84, 'bounds') else 'N/A'}")
    # print(f"inter_wgs84.is_empty: {inter_wgs84.is_empty if hasattr(inter_wgs84, 'is_empty') else 'N/A'}")
    
    # print("\n=== 步骤3：格网生成（按经纬度固定步长） ===")
    grids = generate_lonlat_grid(inter_wgs84, grid_step_deg)
    print("Progress:20%")
    # print(f"生成格网数量: {len(grids)} (步长={grid_step_deg}°)")

    # print("\n=== 步骤4-5：分块裁剪与局部配准 ===")
    matcher = FastMapGlueMatcher(model_path, num_keypoints)
    

    wgs84_to_img1 = poly_projector(img1["crs"])  # WGS84 -> CRS1
    wgs84_to_img2 = poly_projector(img2["crs"])  # WGS84 -> CRS2

    # print("=" * 60)
    # print("诊断控制点收集前状态")
    
    # 检查影像数据
    # print(f"img1['data'] shape: {img1['data'].shape}, dtype: {img1['data'].dtype}")
    # print(f"img1['data'] 范围: [{img1['data'].min()}, {img1['data'].max()}]")
    # print(f"img1['bounds']: {img1['bounds']}")
    # print(f"img1['transform']: {img1['transform']}")
    
    # print(f"img2['data'] shape: {img2['data'].shape}, dtype: {img2['data'].dtype}")
    # print(f"img2['data'] 范围: [{img2['data'].min()}, {img2['data'].max()}]")
    # print(f"img2['bounds']: {img2['bounds']}")
    
    # 检查格网
    # print(f"生成格网数量: {len(grids)}")
    # if len(grids) > 0:
    #     print(f"前3个格网中心: {grids[:3]}")
    #     # 测试坐标转换
    #     test_lon, test_lat = grids[0]
    #     x1, y1 = wgs84_to_img1(test_lon, test_lat)
    #     x2, y2 = wgs84_to_img2(test_lon, test_lat)
    #     print(f"格网0 ({test_lon}, {test_lat}) -> img1: ({x1}, {y1}), img2: ({x2}, {y2})")
        
    #     # 检查是否在影像范围内
    #     print(f"  img1范围内? {img1['bounds'][0] <= x1 <= img1['bounds'][2] and img1['bounds'][1] <= y1 <= img1['bounds'][3]}")
    #     print(f"  img2范围内? {img2['bounds'][0] <= x2 <= img2['bounds'][2] and img2['bounds'][1] <= y2 <= img2['bounds'][3]}")

    P1, P2 = collect_control_points(
        matcher,
        grids,
        img1,
        img2,
        wgs84_to_img1,
        wgs84_to_img2,
        bbox_to_window,
        crop_block,
    )

    if P1.shape[0] < 10:
        raise RuntimeError(f"控制点过少：{P1.shape[0]}。请增大网格、检查影像或模型参数。")
    # print(f"累计控制点数: {P1.shape[0]}")

    # 保存控制点为GeoJSON
    # if save_control_points:
    #     base_dir = os.path.dirname(output_path)
    #     geojson1_path = os.path.join(base_dir, "reference_image.geojson")
    #     geojson2_path = os.path.join(base_dir, "target_image.geojson")
        
        # 先保存所有控制点
        # save_points_as_geojson(P1, img1, geojson1_path)
        # save_points_as_geojson(P2, img2, geojson2_path)

    # print("\n=== 步骤6：控制点整合 & 全局仿射估计 ===")
    M, inliers = estimate_affine_global(P1, P2)
    if M is None:
        raise RuntimeError("仿射矩阵估计失败。")
    # used = int(inliers.sum()) if inliers is not None else 0
    # print("仿射矩阵:\n", M)
    # print(f"内点: {used}/{P1.shape[0]}")

    # 保存内点控制点为GeoJSON
    # if save_control_points and inliers is not None:
    #     inlier_mask = inliers.ravel().astype(bool)
        # P1_inliers = P1[inlier_mask]
        # P2_inliers = P2[inlier_mask]
        
        # geojson1_inliers_path = os.path.join(base_dir, "reference_image_inliers.geojson")
        # geojson2_inliers_path = os.path.join(base_dir, "target_image_inliers.geojson")
        
        # save_points_as_geojson(P1_inliers, img1, geojson1_inliers_path)
        # save_points_as_geojson(P2_inliers, img2, geojson2_inliers_path)

    # print("\n=== 步骤7：全局配准 & 写出 ===")
    out = warp_affine_to_reference(img2, M, img1, output_path)
    print(f"registration success")
    print("Progress:35%")
    
    # print("【配准原始数据维度】")
    # print(f"img1（配准后）原始shape: {img1['data'].shape}")
    # print(f"out（配准后）原始shape: {out['image'].shape}")
    
    standardized_img1 = {
        # 图像数组（保留）
        "image": img1["image"],
        # Affine 转 普通元组（PyTorch 支持）
        "transform": (
            img1["transform"].c,
            img1["transform"].a,
            img1["transform"].b,
            img1["transform"].f,
            img1["transform"].d,
            img1["transform"].e
        ),
        # CRS 转 纯字符串（彻底干掉 rasterio 对象）
        "crs": img1["crs"].to_wkt() if hasattr(img1["crs"], 'to_wkt') else str(img1["crs"]),
        "height": img1["height"],
        "width": img1["width"]
    }

    # out 也强制标准化（防止里面有GIS对象）
    standardized_out = {
        "image": out["image"],
        "transform": (
            out["transform"].c, out["transform"].a, out["transform"].b,
            out["transform"].f, out["transform"].d, out["transform"].e
        ) if hasattr(out["transform"], 'a') else out["transform"],
        "crs": out["crs"].to_wkt() if hasattr(out["crs"], 'to_wkt') else str(out["crs"]),
        "height": out["height"],
        "width": out["width"]
    }
    
    # print("【标准化后（正确维度 C,H,W）】")
    # print(f"standardized_img1 shape: {standardized_img1['image'].shape}")
    # print(f"standardized_out shape: {standardized_out['image'].shape}")
        # ====================================================================================
    
    # 返回 100% 纯Python原生字典（适配 crop + 适配 PyTorch）
    return standardized_out, standardized_img1

    # 生成棋盘格图像

    # if generate_chessboard:
    #     generate_chessboard_with_fix(
    #         img1=img1,
    #         output_path=output_path,
    #         P1=P1,
    #         P2=P2,
    #         chessboard_path="custom_chessboard.tif",
    #         generate_chessboard=True,
    #         patch_size=96  # 自定义patch大小
    #     )
    #     print(f"棋盘格图像已生成: {chessboard_path}")


if __name__ == "__main__":
    # 示例参数（请自行替换为实际路径）
    
    reference_image = my_base_path+"/sample_test/airport_post.tif"
    target_image = my_base_path+"/sample_test/airport_pre.tif"
    output_image = my_base_path+"/sample_test/test.tif"

    out = register_images(
        reference_image,
        target_image,
        output_image,
        grid_step_deg=0.001,   # 经纬度网格步长
        model_path='./weights/fastmapglue_model.pt',
        num_keypoints=1024,
        save_control_points=True,
        generate_chessboard=False,
        chessboard_path=my_base_path+"/affine/chessboard.tif"
        )
