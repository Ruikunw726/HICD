# -*- coding: utf-8 -*-
"""
Created on Wed May 14 22:01:52 2025

@author: elenstian
"""
import os
import numpy as np
import imageio

def batch_convert_folder(input_folder, output_folder, 
                        ori_label_value_dict, target_label_value_dict):
    """
    批量转换文件夹中的标签图像
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param ori_label_value_dict: 原始颜色字典
    :param target_label_value_dict: 目标标签字典
    """
    # 确保输出目录存在[3,6](@ref)
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有PNG文件路径[5](@ref)
    png_files = [f for f in os.listdir(input_folder) 
                if f.lower().endswith('.tif')]
    
    # 创建标签映射字典[6](@ref)
    target_to_ori = {v: k for k, v in target_label_value_dict.items()}
    
    for filename in png_files:
        try:
            # 构建完整路径[3](@ref)
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 读取标签图像（保持原有逻辑）
            labels = np.squeeze(np.array(imageio.imread(input_path), np.float32))
            
            labels[labels == 150] = 2
            labels[labels == 255] = 1
            
            labels[labels==0]=255
            
            # 执行颜色映射转换（原有函数优化）
            H, W = labels.shape
            color_mapped = np.zeros((H, W, 3), dtype=np.uint8)
            
            for target_label, ori_label in target_to_ori.items():
                mask = labels == target_label
                color_mapped[mask] = ori_label_value_dict[ori_label]
            
            # 保存处理结果[6](@ref)
            imageio.imsave(output_path, color_mapped)
            print(f"Converted: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

# 使用示例
if __name__ == "__main__":
    ori_label_value_dict = {
        'background': (0, 0, 0),
        'no_damage': (70, 181, 121),
        'minor_damage': (167, 187, 27),
        'major_damage': (228, 189, 139),
        'destroy': (181, 70, 70)
    }

    target_label_value_dict = {
        'background': 0,
        'no_damage': 1,
        'minor_damage': 2,
        'major_damage': 3,
        'destroy': 4,
    }

    # 设置输入输出路径
    input_dir = r"E:/MambaCD/data_test_plane/test/label/clf"
    output_dir = r"E:/MambaCD/clf_visual"
    
    # 执行批量转换
    batch_convert_folder(input_dir, output_dir,
                        ori_label_value_dict, target_label_value_dict)