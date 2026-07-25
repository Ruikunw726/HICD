from typing import List, Dict
from shapely.geometry import box, Polygon
# from pyproj import Transformer, CRS

from .gdal_proj import CRS, Transformer


def bounds_to_wgs84(bounds, crs) -> Polygon:
    """将任意 CRS 的 bounds 转到 WGS84，返回 shapely Polygon。
    
    如果输入没有 CRS，则自动假定为 WGS84。
    """
    if crs is None:
        print("⚠️ 警告: 输入影像没有 CRS，已自动假定为 EPSG:4326 (WGS84)")
        crs = CRS.from_epsg(4326)
    else:
        crs = CRS.from_user_input(crs)

    dst = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs, dst, always_xy=True)

    minx, miny, maxx, maxy = bounds
    xs = [minx, maxx, maxx, minx]
    ys = [miny, miny, maxy, maxy]

    lon, lat = transformer.transform(xs, ys)
    return Polygon(zip(lon, lat)).envelope


def wgs84_box_to_crs(poly: Polygon, crs) -> Polygon:
    """将 WGS84 的 box/Polygon 转换到目标 CRS。"""
    if crs is None:
        print("⚠️ 警告: 目标 CRS 未定义，已默认使用 EPSG:4326")
        crs = CRS.from_epsg(4326)
    else:
        crs = CRS.from_user_input(crs)

    src = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(src, crs, always_xy=True)

    xs, ys = poly.exterior.coords.xy
    X, Y = transformer.transform(xs, ys)
    return Polygon(zip(X, Y)).envelope


def common_area_wgs84(img1_info: Dict, img2_info: Dict) -> Polygon:
    """计算两图公共区域（在 WGS84 下的 bbox）。"""
    p1 = bounds_to_wgs84(img1_info["bounds"], img1_info["crs"])
    p2 = bounds_to_wgs84(img2_info["bounds"], img2_info["crs"])

    inter = p1.intersection(p2)
    if inter.is_empty:
        print("⚠️ 警告: 两幅影像在 WGS84 下没有交集")
        return None
    return inter.envelope


def generate_lonlat_grid(common_wgs84: Polygon, step_deg: float) -> List[Polygon]:
    """在公共区域内按经纬度步长生成网格（小 bbox 多边形集合，WGS84）。"""
    minx, miny, maxx, maxy = common_wgs84.bounds
    grids: List[Polygon] = []

    x = minx
    while x < maxx:
        y = miny
        nx = min(x + step_deg, maxx)
        while y < maxy:
            ny = min(y + step_deg, maxy)
            cell = box(x, y, nx, ny)
            if cell.intersects(common_wgs84):
                grids.append(cell)
            y += step_deg
        x += step_deg

    # print(f"共生成 {len(grids)} 个格网")
    return grids
