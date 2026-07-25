# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 12:13:47 2025

@author: elenstian
"""

import os
import shutil

# 配置路径和前缀
source_folder = r'E:\MambaCD\data_output\Mariupol_ship\images'  # 替换为你的源文件夹路径
dest_folder = r'E:\MambaCD\data_output\Mariupol_ship_post'    # 替换为目标文件夹路径
prefix = 'Mariupol_ship_'                      # 替换为你需要的前缀

# 确保目标文件夹存在
os.makedirs(dest_folder, exist_ok=True)

# 处理文件
for filename in os.listdir(source_folder):
    if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
        src_path = os.path.join(source_folder, filename)
        new_name = prefix + filename
        dest_path = os.path.join(dest_folder, new_name)
        
        # 复制文件（保留原始文件）
        shutil.copy2(src_path, dest_path)
        print(f'已复制: {filename} -> {new_name}')

print("\n操作完成！所有TIFF文件已添加前缀并安全复制到目标文件夹。原始文件保持不变。")