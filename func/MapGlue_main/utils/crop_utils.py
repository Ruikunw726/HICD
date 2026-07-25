from typing import Tuple, Dict
import numpy as np
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import rowcol


def bbox_to_window(bbox_poly: Polygon, img_info: Dict) -> rasterio.windows.Window:
    """将目标 CRS 下的 bbox 多边形转换为像素窗口。假设 bbox_poly 已在 img_info 的 CRS 中。"""
    minx, miny, maxx, maxy = bbox_poly.bounds
    transform = img_info["transform"]
    col_min, row_max = ~transform * (minx, miny)
    col_max, row_min = ~transform * (maxx, maxy)
    col_min = int(max(0, np.floor(col_min)))
    col_max = int(min(img_info["width"], np.ceil(col_max)))
    row_min = int(max(0, np.floor(row_min)))
    row_max = int(min(img_info["height"], np.ceil(row_max)))
    if col_max <= col_min or row_max <= row_min:
        return None
    return rasterio.windows.Window.from_slices((row_min, row_max), (col_min, col_max))


def crop_block(img_info: Dict, window: rasterio.windows.Window) -> Tuple[np.ndarray, Tuple[int,int]]:
    """按像素窗口裁剪 (返回 HWC 或 HW)。同时返回窗口左上角像素偏移 (col_off, row_off)。"""
    if window is None:
        return None, (0, 0)
    data = img_info["data"]
    # print(data.shape)
    r0, c0 = int(window.row_off), int(window.col_off)
    r1 = r0 + int(window.height)
    c1 = c0 + int(window.width)
    if data.ndim == 3:
        block = data[:, r0:r1, c0:c1]
        block = np.transpose(block, (1, 2, 0))  # HWC
    else:
        block = data[r0:r1, c0:c1]
    return block, (c0, r0)