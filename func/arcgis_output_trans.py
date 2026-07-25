# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 16:23:26 2025

@author: elenstian
"""

import os
import random
import shutil
# import imageio # 不再需要 imageio
# from osgeo import gdal, osr # 不再需要 GDAL/OSR 进行像素操作
import numpy as np # 仍可能用于一些辅助逻辑，但不是直接读取图像

BASE_DATA_PATH = "E:/MambaCD"
SRC_PATH = f"{BASE_DATA_PATH}/data_output/Vuhledar"
TEST_DATA_PATH = f"{BASE_DATA_PATH}/data_test_all"
PREFIX = "Vuhledar_"


# --- 核心文件复制函数 (替换了之前的 GDAL 处理逻辑) ---
def copy_tif_file_with_prefix(input_path: str, output_path: str):
    """
    直接复制 GeoTIFF 文件到指定路径，并确保目标目录存在。
    此函数不读取或修改图像数据，仅进行文件复制。

    Args:
        input_path (str): 输入 GeoTIFF 文件的完整路径。
        output_path (str): 输出 GeoTIFF 文件的完整路径。
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在
    
    try:
        shutil.copy2(input_path, output_path) # 使用 shutil.copy2 保留更多元数据
        print(f"成功复制: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
    except Exception as e:
        print(f"复制失败: {os.path.basename(input_path)} - {str(e)}")

# --- 主执行函数 ---
def main():
    copy_all_datasets_to_tif() # 步骤1: 复制所有数据到统一 TIFF 目录
    split_all_datasets()       # 步骤2: 划分所有数据集
    save_dataset_lists()       # 步骤3: 保存文件列表

# 复制图像和标签函数
def copy_all_datasets_to_tif():
    """
    遍历源目录下的 TIFF 文件，将其复制到目标目录，并添加指定前缀。
    不再进行任何图像数据处理或格式转换。
    """
    # 复制图像
    for img_type, src_subpath in [("pre", "images2"), ("post", "images")]:
        input_folder = f"{SRC_PATH}/{src_subpath}"
        output_folder = f"{TEST_DATA_PATH}/all_image/{img_type}"
        os.makedirs(output_folder, exist_ok=True) # 确保输出文件夹存在

        for filename in os.listdir(input_folder):
            if not filename.lower().endswith((".tif", ".tiff")):
                continue # 只处理 .tif 文件

            input_path = os.path.join(input_folder, filename)
            # 输出文件名保持原有名称，添加前缀，并保留 .tif 后缀
            output_name = PREFIX + os.path.splitext(filename)[0] + ".tif"
            output_path = os.path.join(output_folder, output_name)

            copy_tif_file_with_prefix(input_path, output_path)

    # 复制标签 (clf 和 loc 共享相同的源)
    for label_type in ["clf", "loc"]:
        input_folder = f"{SRC_PATH}/labels"
        output_folder = f"{TEST_DATA_PATH}/all_label/{label_type}"
        os.makedirs(output_folder, exist_ok=True) # 确保输出文件夹存在

        for filename in os.listdir(input_folder):
            if not filename.lower().endswith((".tif", ".tiff")):
                continue # 只处理 .tif 文件

            input_path = os.path.join(input_folder, filename)
            # 输出文件名保持原有名称，添加前缀，并保留 .tif 后缀
            output_name = PREFIX + os.path.splitext(filename)[0] + ".tif"
            output_path = os.path.join(output_folder, output_name)

            copy_tif_file_with_prefix(input_path, output_path)

# 划分数据集函数
def split_all_datasets():
    # 需要划分的数据集类型及其对应路径
    datasets = {
        "image/pre": ("all_image/pre", "train/image/pre", "test/image/pre"),
        "image/post": ("all_image/post", "train/image/post", "test/image/post"),
        "label/clf": ("all_label/clf", "train/label/clf", "test/label/clf"),
        "label/loc": ("all_label/loc", "train/label/loc", "test/label/loc")
    }

    for dataset_type, (src, train_dst, test_dst) in datasets.items():
        # 注意: PREFIX 已经在 copy_all_datasets_to_tif 中添加，
        # 因此这里的 prefix 参数保持为空字符串，避免重复添加前缀。
        split_dataset_with_prefix(
            src_folder=f"{TEST_DATA_PATH}/{src}",
            train_folder=f"{TEST_DATA_PATH}/{train_dst}",
            test_folder=f"{TEST_DATA_PATH}/{test_dst}",
            prefix="", # 文件名已包含前缀，这里不再添加
            split_ratio=0.8,
            seed=42,
            move=False
        )

# 保存文件名列表
def save_dataset_lists():
    # 保存训练集和测试集文件列表
    save_tif_names( # 调用新的保存函数
        input_dir=f"{TEST_DATA_PATH}/train/image/pre",
        output_file=f"{TEST_DATA_PATH}/train/train.txt"
    )

    save_tif_names( # 调用新的保存函数
        input_dir=f"{TEST_DATA_PATH}/test/image/pre",
        output_file=f"{TEST_DATA_PATH}/test/test.txt"
    )

def save_tif_names(input_dir: str, output_file: str):
    """
    获取指定目录下所有 TIFF 文件的文件名（不带扩展名），并保存到 TXT 文件中。

    Args:
        input_dir (str): 包含 TIFF 文件的目录。
        output_file (str): 输出 TXT 文件的完整路径。
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 获取所有 TIFF 文件名（不区分大小写）
    tif_files = [
        os.path.splitext(f)[0]
        for f in os.listdir(input_dir)
        if f.lower().endswith((".tif", ".tiff"))
    ]

    # 写入 TXT 文件（无扩展名）
    with open(output_file, 'w') as f:
        for filename in tif_files:
            f.write(f"{filename}\n")

def split_dataset_with_prefix(
    src_folder: str,
    train_folder: str,
    test_folder: str,
    prefix: str = "",
    split_ratio: float = 0.8,
    seed: int = 42,
    move: bool = False
):
    """
    划分数据集并添加前缀（如果需要）。现在支持 TIFF 文件。

    Args:
        src_folder: 源文件夹路径。
        train_folder: 训练集保存路径。
        test_folder: 测试集保存路径。
        prefix: 文件名前缀。
        split_ratio: 训练集比例。
        seed: 随机种子。
        move: 是否移动文件（默认False，即复制）。
    """
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 获取所有 TIFF 文件（不递归子文件夹）
    all_files = [
        f for f in os.listdir(src_folder)
        if f.lower().endswith((".tif", ".tiff")) # 修改为检查 .tif 后缀
    ]

    random.seed(seed)
    random.shuffle(all_files)

    split_idx = int(len(all_files) * split_ratio)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    copy_func = shutil.move if move else shutil.copy

    def _process_files(files, dest_dir):
        for f in files:
            src_path = os.path.join(src_folder, f)
            new_name = f"{prefix}{f}" # 添加前缀 (如果 prefix 非空)
            dst_path = os.path.join(dest_dir, new_name)
            copy_func(src_path, dst_path)

    _process_files(train_files, train_folder)
    _process_files(test_files, test_folder)

    print(f"处理完成（{'移动' if move else '复制'}模式）")
    print(f"源文件夹: {src_folder}")
    print(f"总文件数: {len(all_files)}")
    print(f"训练集: {len(train_files)} 文件 → {train_folder}")
    print(f"测试集: {len(test_files)} 文件 → {test_folder}")

# 使用示例
if __name__ == "__main__":
    main()