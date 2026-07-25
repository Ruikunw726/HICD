from typing import List, Tuple, Dict
import numpy as np
import cv2
import torch
from shapely.geometry import Polygon
from tqdm import tqdm

# from .io_utils import to_uint8_for_cv

from .io_utils import to_uint8_for_cv, read_geotiff
from .warp_utils import build_gcps_and_warp



class FastMapGlueMatcher:
    def __init__(self, model_path: str = './weights/fastmapglue_model.pt', num_keypoints: int = 1024):
        self.num_keypoints = num_keypoints
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path: str):
        try:
            self.model = torch.jit.load(model_path)
            self.model.eval()
            # print("FastMapGlue 模型加载成功!")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")

    @torch.no_grad()
    def match_block(self, img0: np.ndarray, img1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """对两块(HWC)图像进行匹配，返回像素坐标对 (N,2)。"""
        if img0 is None or img1 is None:
            return None, None
        if min(img0.shape[0], img0.shape[1], img1.shape[0], img1.shape[1]) < 32:
            return None, None

        # 转 uint8 RGB
        im0 = to_uint8_for_cv(img0)
        im1 = to_uint8_for_cv(img1)

        t0 = torch.from_numpy(im0)
        t1 = torch.from_numpy(im1)
        n = torch.tensor(int(self.num_keypoints))

        try:
            pts = self.model(t0, t1, n)  # (N,4) -> [x0, y0, x1, y1]
            if pts is None or pts.numel() == 0:
                return None, None
            pts = pts.cpu().numpy()
            p0 = pts[:, :2]
            p1 = pts[:, 2:]
        except RuntimeError as e:
            print(f"匹配失败: {e}")
            return None, None

        # RANSAC 去外点（单应/基础矩阵均可，默认单应）
        if len(p0) >= 4:
            H, inlier = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, ransacReprojThreshold=3, maxIters=10000, confidence=0.9999)
            if inlier is not None and inlier.sum() >= 4:
                inlier = inlier.ravel().astype(bool)
                p0 = p0[inlier]
                p1 = p1[inlier]
        return p0, p1


def collect_control_points(
    matcher: FastMapGlueMatcher,
    grids_wgs84: List[Polygon],
    img1_info: Dict,
    img2_info: Dict,
    wgs84_to_img1,
    wgs84_to_img2,
    bbox_to_window_fn,
    crop_block_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    """对每个网格做裁剪+局部匹配，汇总到全局像素坐标。
    返回 points1, points2 (全局像素坐标，均以各自原图像素系)。
    """
    all_p1 = []
    all_p2 = []

    total_cells = len(grids_wgs84)
    # for cell in tqdm(grids_wgs84, desc="格网处理", unit="cell"):
    for idx,cell in enumerate(grids_wgs84):
        if idx % max(1,total_cells//20)==0:
            print(f"processing grids:{idx}/{total_cells}({100*idx/total_cells:.1f})%")
        # 将 WGS84 bbox 投影到各自图像 CRS
        cell1 = wgs84_to_img1(cell)
        cell2 = wgs84_to_img2(cell)

        # 转像素窗口并裁剪
        win1 = bbox_to_window_fn(cell1, img1_info)
        win2 = bbox_to_window_fn(cell2, img2_info)
        if win1 is None or win2 is None:
            continue
            
        # 检查图像块质量
        blk1, (c1, r1) = crop_block_fn(img1_info, win1)
        blk2, (c2, r2) = crop_block_fn(img2_info, win2)
        if blk1 is None or blk2 is None:
            continue
            
        # 检查图像块对比度
        if np.std(blk1) < 1e-3 or np.std(blk2) < 1e-3:
            continue

        p0, p1 = matcher.match_block(blk1, blk2)
        if p0 is None or p1 is None or len(p0) < 4:  # 修改这里：要求至少4个点
            continue

        # 局部坐标 -> 全局像素坐标
        g0 = p0 + np.array([c1, r1], dtype=np.float32)
        g1 = p1 + np.array([c2, r2], dtype=np.float32)
        all_p1.append(g0)
        all_p2.append(g1)

    if len(all_p1) == 0:
        return np.empty((0,2), dtype=np.float32), np.empty((0,2), dtype=np.float32)
    P1 = np.concatenate(all_p1, axis=0)
    P2 = np.concatenate(all_p2, axis=0)
    return P1, P2


def estimate_affine_global(points_ref: np.ndarray, points_tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """用 RANSAC 估计全局仿射，使得 tgt→ref。返回 (2x3 矩阵, inlier 掩码)。"""
    if points_ref.shape[0] < 3:
        return None, None
    M, inliers = cv2.estimateAffine2D(points_tgt, points_ref, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=10000, confidence=0.99)
    return M, inliers


# def register_images(img1_path, img2_path, tiepoints, out_path):
#     """
#     使用 TPS 方法进行配准（替换 estimate_affine_global）
#     :param img1_path: 参考影像
#     :param img2_path: 待配准影像
#     :param tiepoints: [(x_img2, y_img2, x_img1, y_img1), ...] 同名点对
#     :param out_path: 输出路径
#     """
#     build_gcps_and_warp(img2_path, out_path, tiepoints, dst_srs="EPSG:4326")
#     return out_path
