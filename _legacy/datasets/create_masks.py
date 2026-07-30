# conda activate LDJ
# pip install scikit-image
# 从.json文件变成模型所需的.png文件

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
np.random.seed(1)
import random
random.seed(1)
import pandas as pd
import cv2
import timeit
from os import path, makedirs, listdir
import sys
sys.setrecursionlimit(10000)
from multiprocessing import Pool
from skimage.morphology import square, dilation, watershed, erosion
from skimage import io

from shapely.wkt import loads
from shapely.geometry import mapping, Polygon

# import matplotlib.pyplot as plt
# import seaborn as sns

import json

masks_dir = 'masks' # 保存生成的掩码图像的目录(按需更改)

train_dirs = ['train', 'tier3'] # 包含训练数据的目录列表（按需更改）


# 将多边形转换为掩码图像
def mask_for_polygon(poly, im_size=(1024, 1024)): # 多边形对象，掩码图像大小
    img_mask = np.zeros(im_size, np.uint8) # 创建一个全零的二维数组，形状为 im_size，数据类型为 np.uint8。这个数组将作为掩码图像的初始值
    int_coords = lambda x: np.array(x).round().astype(np.int32) # 定义一个 lambda 函数 int_coords，用于将浮点坐标转换为整数坐标。这一步是必要的，因为 OpenCV 的 fillPoly 函数需要整数坐标
    exteriors = [int_coords(poly.exterior.coords)] # 获取多边形的外轮廓坐标（整数）
    interiors = [int_coords(pi.coords) for pi in poly.interiors] # 获取多边形的内轮廓（孔）坐标（整数）
    cv2.fillPoly(img_mask, exteriors, 1) # 使用 cv2.fillPoly 填充外轮廓内的区域为 1
    cv2.fillPoly(img_mask, interiors, 0) # 将内轮廓内的区域填充为 0
    return img_mask


# 定义损坏类型的映射字典，将损坏类型映射为整数值
damage_dict = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1 # ?
}

damage_dict = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1 # ?
}


# 处理单个 JSON 文件，生成建筑物掩码和损坏掩码
def process_image(json_file): # Json文件的路径
    js1 = json.load(open(json_file)) # 加载前灾害的 JSON 文件
    js2 = json.load(open(json_file.replace('_pre_disaster', '_post_disaster'))) # 加载后灾害的 JSON 文件，路径通过替换 _pre_disaster 为 _post_disaster 得到

    msk = np.zeros((1024, 1024), dtype='uint8') # 创建两个全零的二维数组，形状为 (1024, 1024)，数据类型为 np.uint8。msk 用于存储建筑物掩码
    msk_damage = np.zeros((1024, 1024), dtype='uint8') # msk_damage 用于存储损坏掩码

    # 处理前灾害数据
    for feat in js1['features']['xy']: # 遍历前灾害 JSON 文件中的每个特征
        poly = loads(feat['wkt']) # 加载特征的几何信息（WKT 格式）
        _msk = mask_for_polygon(poly) # 生成多边形的掩码图像
        msk[_msk > 0] = 255 # 将生成的掩码图像中的非零像素值设置为 255，表示建筑物区域

    # 处理后灾害数据
    for feat in js2['features']['xy']:
        poly = loads(feat['wkt']) # 加载特征的几何信息（WKT 格式）
        subtype = feat['properties']['subtype'] # 获取特征的子类型（损坏类型）
        _msk = mask_for_polygon(poly) # 生成多边形的掩码图像
        msk_damage[_msk > 0] = damage_dict[subtype] # 根据 subtype（损坏类型）从 damage_dict 中获取对应的整数值，并将生成的掩码图像中的非零像素值设置为该值，表示损坏区域。

    # 使用 OpenCV 的 cv2.imwrite 函数保存生成的掩码图像。通过替换路径中的 /labels/ 为 /masks/ 和文件扩展名 .json 为 .png，生成保存路径。使用 cv2.IMWRITE_PNG_COMPRESSION 参数设置 PNG 图像的压缩级别为 9，以减少文件大小。
    cv2.imwrite(json_file.replace('/labels/', '/masks/').replace('_pre_disaster.json', '_pre_disaster.png'), msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(json_file.replace('/labels/', '/masks/').replace('_pre_disaster.json', '_post_disaster.png'), msk_damage, [cv2.IMWRITE_PNG_COMPRESSION, 9])



# 处理所有训练数据，生成建筑物掩码和损坏掩码。
if __name__ == '__main__':
    t0 = timeit.default_timer() # 初始化计时器

    all_files = []
    # 遍历所有训练目录，收集所有前灾害 JSON 文件的路径
    for d in train_dirs:
        makedirs(path.join(d, masks_dir), exist_ok=True)
        for f in sorted(listdir(path.join(d, 'images'))):
            if '_pre_disaster.png' in f:
                all_files.append(path.join(d, 'labels', f.replace('_pre_disaster.png', '_pre_disaster.json')))


    # 使用多线程池 Pool 并行处理所有 JSON 文件
    with Pool() as pool:
        _ = pool.map(process_image, all_files)

    # 计算并打印处理时间
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
    
    # 到此为止，生成的.png文件会和对应的.json文件放在一个目录下，因此，需要运行另一个程序将该目录下的.png文件全部转移到它们应该在的路径