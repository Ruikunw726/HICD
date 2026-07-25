# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 19:47:47 2025

@author: elenstian
"""

import os
import json
import numpy as np
from osgeo import gdal, osr
import cv2
from skimage.measure import label, regionprops # 引入 scikit-image 进行连通域分析
gdal.UseExceptions()

def pixel_to_geo(geo_transform, x_pixel, y_pixel):
    """
    将像素坐标 (x_pixel, y_pixel) 转换为 GeoTIFF 原始坐标系下的地理坐标 (X_geo, Y_geo)。

    参数:
        geo_transform (tuple): GDAL GetGeoTransform() 返回的地理仿射变换元组。
        x_pixel (float): 像素的列坐标 (从左到右，从0开始)。
        y_pixel (float): 像素的行坐标 (从上到下，从0开始)。

    返回:
        tuple: (X_geo, Y_geo) 对应的原始地理坐标。
    """
    origin_x = geo_transform[0]
    origin_y = geo_transform[3]
    pixel_width = geo_transform[1]
    pixel_height = geo_transform[5] # 通常是负值，表示Y轴方向

    geo_x = origin_x + x_pixel * pixel_width + y_pixel * geo_transform[2]
    geo_y = origin_y + x_pixel * geo_transform[4] + y_pixel * pixel_height

    return geo_x, geo_y

# --- 核心函数：提取变化区域的精确外轮廓信息 ---
def get_change_area_contours(image_array, geo_transform, proj_wkt, background_value=0):
    """
    识别图像数组中所有非背景区域（像素值为 1 或 2），提取它们的精确多边形外轮廓点集，
    并将其转换为地理（WGS84）坐标。

    参数:
        image_array (np.ndarray): 变化检测结果图像数组 (H, W)，像素值代表变化类型。
                                  像素值：0=背景，1=未损毁，2=损毁。
        geo_transform (tuple): GeoTIFF 的地理仿射变换参数。
        proj_wkt (str): GeoTIFF 的投影信息（WKT 字符串）。
        background_value (int): 表示背景的像素值（默认为 0）。

    返回:
        list: 包含每个变化区域信息的字典列表，每个字典包含变化类型、像素轮廓点集和地理轮廓点集。
    """
    change_areas_info = []
    
    # 设置坐标转换对象（只创建一次以提高效率）
    source_srs = osr.SpatialReference()
    target_srs = osr.SpatialReference()
    coord_transform = None

    try:
        source_srs.ImportFromWkt(proj_wkt)
        target_srs.ImportFromEPSG(4326) # EPSG:4326 代表 WGS84 经纬度
        if not source_srs.IsSame(target_srs):
            coord_transform = osr.CoordinateTransformation(source_srs, target_srs)
        elif not source_srs.IsGeographic():
             print(f"警告: 源坐标系是地理坐标系但不是 WGS84，或为投影坐标系。如果未发生转换，请确保 proj_wkt 对于直接地理转换有效。")
    except Exception as e:
        print(f"错误：设置坐标转换失败: {e}")
        print("请检查 GeoTIFF 投影信息（WKT）的有效性。地理坐标可能不准确。")
        return [] # 无法进行可靠的地理转换，返回空列表


    # 遍历我们想要提取的每个类别（1：未损毁，2：损毁）
    for class_id in [1, 2]:
        # 为当前类别创建一个二值掩膜
        # OpenCV 的 findContours 要求输入图像为 np.uint8 类型，且值为 0 或 255
        binary_mask = np.uint8(image_array == class_id) * 255

        # 使用 OpenCV 查找轮廓
        # RETR_EXTERNAL 仅检索外部轮廓（忽略内部孔洞）
        # CHAIN_APPROX_SIMPLE 压缩轮廓点，减少冗余，同时保持形状
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            # 轮廓简化（可选，但推荐，平衡精度与数据量）
            # epsilon 是近似的阈值。epsilon 越大，点越少，轮廓越粗糙。
            # 这里使用一个较小的 epsilon 值，保留更多细节。
            epsilon = 0.0001 * cv2.arcLength(cnt, True) # 调整此值以控制轮廓点的密度
            approx_contour_pixels = cv2.approxPolyDP(cnt, epsilon, True)

            # 挤压维度并转换为 Python 列表 [[x1, y1], [x2, y2], ...]
            pixel_contour_points = approx_contour_pixels.squeeze().tolist()

            # 处理单个点轮廓的情况 (例如，approx 返回的不是 [[x,y]] 而是 [x,y])
            if not isinstance(pixel_contour_points[0], list):
                 pixel_contour_points = [pixel_contour_points]


            # 根据 class_id 确定变化类型
            change_type = ""
            if class_id == 1:
                change_type = "未损毁"
            elif class_id == 2:
                change_type = "损毁"
            else:
                # 理论上不会执行到这里，因为 class_id 只会是 1 或 2
                continue

            # --- 将像素轮廓点转换为地理坐标 ---
            geo_contour_points = []
            for p_x, p_y in pixel_contour_points:
                # 转换为 GeoTIFF 原始投影系统中的地理坐标
                orig_geo_x, orig_geo_y = pixel_to_geo(geo_transform, p_x, p_y)

                # 如果需要转换，则转换为 WGS84 经纬度
                if coord_transform:
                    try:
                        # TransformPoint 返回 (经度, 纬度, 高度)，我们只需要经纬度
                        lon, lat, _ = coord_transform.TransformPoint(orig_geo_x, orig_geo_y)
                        geo_contour_points.append([lon, lat])
                    except Exception as e:
                        print(f"警告: 轮廓点 ({p_x},{p_y}) 的坐标转换失败: {e}。将跳过此点。")
                        # 如果单个点转换失败，可以考虑跳过整个轮廓，或者只包含成功转换的点
                        continue
                else:
                    # 如果不需要转换，假设原始坐标已经是 WGS84 经纬度
                    geo_contour_points.append([orig_geo_x, orig_geo_y])
            
            # 如果转换后没有有效的地理点，则不添加此区域
            if not geo_contour_points:
                print(f"警告: 轮廓在地理坐标转换后无有效点，跳过此区域。")
                continue

            change_areas_info.append({
                "变化类型": change_type,
                "变化区域像素坐标": pixel_contour_points, # 轮廓点集：[[x1,y1], [x2,y2], ...]
                "变化区域经纬度坐标": geo_contour_points  # 轮廓点集：[[lon1,lat1], [lon2,lat2], ...]
            })
    
    return change_areas_info

def convert_tif_to_json(tif_path: str, output_json_path: str):
    """
    将 GeoTIFF 变化检测结果文件转换为指定的 JSON 格式。
    它识别所有非背景区域（像素值 1 或 2），并提供其精确外轮廓点集（像素和地理）。

    参数:
        tif_path (str): GeoTIFF 变化检测结果文件的路径。
                        预期像素值：0=背景，1=未损毁，2=损毁。
        output_json_path (str): 输出 JSON 文件的保存路径。
    """

    dataset = gdal.Open(tif_path, gdal.GA_ReadOnly)

    if dataset is None:
        return

    image_array = dataset.ReadAsArray()

    # 确保 image_array 是 2D (H, W)，便于 OpenCV 处理
    if image_array.ndim == 3 and image_array.shape[0] == 1:
        image_array = np.squeeze(image_array)
    elif image_array.ndim == 3 and image_array.shape[0] > 1:
        image_array = image_array[0, :, :] # 取第一个波段
    elif image_array.ndim != 2:
        dataset = None
        return

    # 转换为 unsigned 8-bit integer，OpenCV 的 findContours 函数要求此数据类型
    image_array = image_array.astype(np.uint8)

    geo_transform = dataset.GetGeoTransform()
    proj_wkt = dataset.GetProjection()
    height, width = image_array.shape # 获取挤压后的维度

    dataset = None # 关闭数据集，释放资源

    base_name = os.path.splitext(os.path.basename(tif_path))[0]

    # 提取所有非背景变化区域的精确轮廓信息
    all_change_results = get_change_area_contours(image_array, geo_transform, proj_wkt, background_value=0)

    # 构建最终的 JSON 结构
    json_data = {
        "影像名称": {
            "前一时相": f"{base_name}_pre",
            "后一时相": f"{base_name}_post"
        },
        "固定设施变化检测结果": all_change_results
    }

    # 确保输出目录存在
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"错误: 无法创建输出目录 {output_dir}: {e}")
            return

    # 将数据写入 JSON 文件
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2) # indent=2 提高可读性
    except Exception as e:
        print(f"写入 JSON 文件失败: {e}")
        

def tif_json(input_folder,output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取文件夹中的所有TIFF文件
    tif_files = [f for f in os.listdir(input_folder) 
                if f.lower().endswith(('.tif', '.tiff'))]
    
    # 使用进度条显示转换进度
    for filename in tif_files:
        input_tif_file = os.path.join(input_folder, filename)
        
        # 创建对应的输出JSON文件名（保持原始文件名，只修改扩展名）
        json_filename = os.path.splitext(filename)[0] + ".json"
        output_json_file = os.path.join(output_folder, json_filename)
    
        
        # 运行转换
        convert_tif_to_json(input_tif_file, output_json_file)
        


        
if __name__ == "__main__":

    
    type1 = 1
    
    if type1 == 0:
        input_tif_file = "Kartoum_airport01_000000000003.tif"
        output_json_file = "all_change_results.json"
    
        # 运行转换
        convert_tif_to_json(input_tif_file, output_json_file)
    
        if os.path.exists(output_json_file):
            print("\n--- 生成的 JSON 文件内容预览 ---")
            with open(output_json_file, 'r', encoding='utf-8') as f:
                print(f.read())
            print("--------------------------------")
    else:
        # 设置输入文件夹和输出文件夹路径
        input_folder = r"C:\Users\elenstian\Desktop\模型训练结果展示 - 副本\战损检测运行结果tif"  # 替换为你的TIFF文件夹路径
        output_folder = r"C:\Users\elenstian\Desktop\模型训练结果展示 - 副本\战损检测运行结果json1"  # 替换为你的JSON输出文件夹路径
        
        tif_json(input_folder,output_folder)
        
        
        


