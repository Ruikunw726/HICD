# -*- coding: utf-8 -*-
"""

使用GDAL将单通道TIFF标签文件转换为彩色可视化PNG图像

"""

import os
import numpy as np
from osgeo import gdal

# 配置GDAL异常处理
gdal.UseExceptions()

def tiff_to_visual_png(input_tif, output_png):
    """
    将单通道TIFF标签文件转换为彩色可视化PNG
    :param input_tif: 输入TIFF文件路径
    :param output_png: 输出PNG文件路径
    """
    try:
        # 打开TIFF文件
        ds = gdal.Open(input_tif)
        if ds is None:
            raise ValueError(f"无法打开文件: {input_tif}")
        
        # 获取波段和数组
        band = ds.GetRasterBand(1)
        labels = band.ReadAsArray().astype(np.uint8)
        
        # 应用预处理转换规则
        labels[labels == 150] = 2    # minor_damage
        labels[labels == 255] = 1    # no_damage
        labels[labels == 0] = 255    # 预处理转换
        
        # 创建RGB输出数组
        height, width = labels.shape
        rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 定义颜色映射 (与之前代码相同)
        color_map = {
            0: [0, 0, 0],          # background - 黑色
            1: [70, 181, 121],     # no_damage - 绿色
            2: [167, 187, 27],     # minor_damage - 黄色
            # 处理之前设置的255值
            255: [0, 0, 0]         # 黑色 (background)
        }
        
        # 应用颜色映射
        for value, color in color_map.items():
            mask = (labels == value)
            for channel in range(3):
                rgb_array[mask, channel] = color[channel]
        
        # 保存为PNG
        driver = gdal.GetDriverByName('PNG')
        out_ds = driver.Create(output_png, width, height, 3, gdal.GDT_Byte)
        
        # 设置地理参考信息（可选）
        out_ds.SetProjection(ds.GetProjection())
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        
        # 写入RGB数据
        for i in range(3):
            out_band = out_ds.GetRasterBand(i+1)
            out_band.WriteArray(rgb_array[:, :, i])
            out_band.FlushCache()
        
        # 清理资源
        out_band = None
        out_ds = None
        ds = None
        
        print(f"转换成功: {os.path.basename(input_tif)} -> {os.path.basename(output_png)}")
        return True
    
    except Exception as e:
        print(f"处理 {input_tif} 时出错: {str(e)}")
        return False

def batch_convert_folder(input_folder, output_folder):
    """
    批量转换文件夹中的所有TIFF文件
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有TIFF文件
    tiff_files = [f for f in os.listdir(input_folder) 
                 if f.lower().endswith(('.tif', '.tiff'))]
    
    # 处理每个文件
    for filename in tiff_files:
        input_path = os.path.join(input_folder, filename)
        # 保留原始文件名，只修改扩展名
        output_name = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(output_folder, output_name)
        
        tiff_to_visual_png(input_path, output_path)

if __name__ == "__main__":
    # 设置输入输出路径
    input_dir = r"E:/MambaCD/data_test_plane/test/label/clf"
    output_dir = r"E:/MambaCD/clf_visual"
    
    # 执行批量转换
    batch_convert_folder(input_dir, output_dir)
    print("批量转换完成!")