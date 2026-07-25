import os
from typing import Tuple
from shapely.geometry import Polygon
from pyproj import CRS, Transformer

from utils.io_utils import read_geotiff
from utils.geo_utils import common_area_wgs84, generate_lonlat_grid, wgs84_box_to_crs
from utils.crop_utils import bbox_to_window, crop_block
from utils.registration import FastMapGlueMatcher, collect_control_points, estimate_affine_global
from utils.warp_utils import warp_affine_to_reference


def make_transformer(src_crs, dst_crs):
    return Transformer.from_crs(CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs), always_xy=True)


def poly_projector(dst_crs):
    """返回一个函数：将 WGS84 多边形投影到 dst_crs。"""
    from utils.geo_utils import wgs84_box_to_crs
    def _fn(poly: Polygon):
        return wgs84_box_to_crs(poly, dst_crs)
    return _fn


def register_images(
    img1_path: str,
    img2_path: str,
    output_path: str,
    grid_step_deg: float = 0.01,
    model_path: str = './weights/fastmapglue_model.pt',
    num_keypoints: int = 1024,
):
    print("=== 步骤1：读取与预处理 ===")
    img1 = read_geotiff(img1_path)
    img2 = read_geotiff(img2_path)

    print("参考影像:", img1_path)
    print("待配准影像:", img2_path)
    print(f"CRS1={img1['crs']}, CRS2={img2['crs']}")

    print("\n=== 步骤2：计算公共区域（WGS84） ===")
    inter_wgs84 = common_area_wgs84(img1, img2)
    if inter_wgs84 is None or inter_wgs84.is_empty:
        raise RuntimeError("两幅影像无重叠区域，无法配准。")
    print("公共区域 bbox:", inter_wgs84.bounds)

    print("\n=== 步骤3：格网生成（按经纬度固定步长） ===")
    grids = generate_lonlat_grid(inter_wgs84, grid_step_deg)
    print(f"生成格网数量: {len(grids)} (步长={grid_step_deg}°)")

    print("\n=== 步骤4-5：分块裁剪与局部配准 ===")
    matcher = FastMapGlueMatcher(model_path, num_keypoints)

    wgs84_to_img1 = poly_projector(img1["crs"])  # WGS84 -> CRS1
    wgs84_to_img2 = poly_projector(img2["crs"])  # WGS84 -> CRS2

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
    print(f"累计控制点数: {P1.shape[0]}")

    print("\n=== 步骤6：控制点整合 & 全局仿射估计 ===")
    M, inliers = estimate_affine_global(P1, P2)
    if M is None:
        raise RuntimeError("仿射矩阵估计失败。")
    used = int(inliers.sum()) if inliers is not None else 0
    print("仿射矩阵:\n", M)
    print(f"内点: {used}/{P1.shape[0]}")

    print("\n=== 步骤7：全局配准 & 写出 ===")
    out = warp_affine_to_reference(img2, M, img1, output_path)
    print(f"配准完成：{out}")


if __name__ == "__main__":
    # 示例参数（请自行替换为实际路径）
    reference_image = "/home/user/Downloads/temp/test_ESRI_18_4326.tif"
    target_image = "/home/user/Downloads/temp/sar_test.tif"
    output_image = "/home/user/Downloads/temp/test.tif"

    register_images(
        reference_image,
        target_image,
        output_image,
        grid_step_deg=0.1,   # 经纬度网格步长
        model_path='./weights/fastmapglue_model.pt',
        num_keypoints=1024,
    )