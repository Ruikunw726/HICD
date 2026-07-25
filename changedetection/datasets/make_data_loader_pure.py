import argparse
import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()


ROOT1 = FILE.parents[1]
ROOT2 = FILE.parents[2]
ROOT3 = FILE.parents[3]

if str(ROOT1) not in sys.path:
    sys.path.append(str(ROOT1))
if str(ROOT2) not in sys.path:
    sys.path.append(str(ROOT2))
if str(ROOT3) not in sys.path:
    sys.path.append(str(ROOT3))

import imageio
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import Counter
from MambaCD.changedetection.datasets.label_change import *
from tqdm import tqdm
# from MambaCD.func.tif_to_json import tif_json
from MambaCD.func.tif_to_xml import tif_xml_single
import torch
from osgeo import gdal

gdal.UseExceptions()

import MambaCD.changedetection.datasets.imutils as imutils


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img

def img_loader_3(path):
    img = np.array(imageio.imread(path), np.float32)
    img = img[:,:,:3]
    return img

def tif_loader_window(path, x_off=0, y_off=0, x_size=None, y_size=None):
    """
    使用 GDAL 读取 GeoTIFF 图像的指定窗口区域
    """
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        print(f"无法打开文件: {path}")
        return None, None, None
    
    # 获取原始图像尺寸
    total_width = dataset.RasterXSize
    total_height = dataset.RasterYSize
    
    # 设置默认窗口大小
    if x_size is None:
        x_size = total_width - x_off
    if y_size is None:
        y_size = total_height - y_off
    
    # 确保窗口不超出图像边界
    x_size = min(x_size, total_width - x_off)
    y_size = min(y_size, total_height - y_off)
    
    # 读取指定窗口的数据
    if dataset.RasterCount >= 3:
        # 读取前三个波段
        band1 = dataset.GetRasterBand(1).ReadAsArray(x_off, y_off, x_size, y_size)
        band2 = dataset.GetRasterBand(2).ReadAsArray(x_off, y_off, x_size, y_size)
        band3 = dataset.GetRasterBand(3).ReadAsArray(x_off, y_off, x_size, y_size)
        # img = np.stack([band1, band2, band3], axis=-1)  # 形状为 (H, W, 3)
        img = np.stack([band3, band2, band1], axis=-1)
        
        
    else:
        # 单波段图像
        img = dataset.GetRasterBand(1).ReadAsArray(x_off, y_off, x_size, y_size)
    
    # 获取地理变换信息（需要调整原点）
    geo_transform = list(dataset.GetGeoTransform())
    geo_transform[0] = geo_transform[0] + x_off * geo_transform[1] + y_off * geo_transform[2]
    geo_transform[3] = geo_transform[3] + x_off * geo_transform[4] + y_off * geo_transform[5]
    
    proj_wkt = dataset.GetProjection()
    dataset = None  # 关闭数据集
    
    img = img.astype(np.float32)
    return img, tuple(geo_transform), proj_wkt
    


def one_hot_encoding(image, num_classes=8):
    # Create a one hot encoded tensor
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    # Move the channel axis to the front
    # one_hot = np.moveaxis(one_hot, -1, 0)

    return one_hot

class DamageDataset_infer(Dataset):
    def __init__(
        self, 
        pre_image,  # 统一参数：路径str 或 crop字典
        post_image, # 统一参数：路径str 或 crop字典
        crop_size=512, 
        max_iters=None, 
        type='train', 
        data_loader=None,
        is_large_image=True,
        patch_size=512, 
        overlap=64, 
        pure_inference=True
    ):
        # ===================== 核心：自动判断输入类型（路径/字典）=====================
        # 输入是字典（crop返回的结果）
        self.is_dict_input = isinstance(pre_image, dict) and isinstance(post_image, dict)
        
        # 1. 字典模式（crop后内存数据）
        if self.is_dict_input:
            self.pre_data = pre_image
            self.post_data = post_image
            self.pre_image_path = None
            self.post_image_path = None
        # 2. 路径模式（原始文件）
        else:
            self.pre_image_path = pre_image
            self.post_image_path = post_image
            self.pre_data = None
            self.post_data = None
        # ==========================================================================

        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.is_large_image = is_large_image
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.crop_size = crop_size
        self.pure_inference = pure_inference
        
        # 数据名
        if self.is_dict_input:
            self.data_list = ["memory_image"]
        else:
            base_name = os.path.splitext(os.path.basename(self.pre_image_path))[0]
            self.data_list = [base_name]
        
        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        
        # 大图分块初始化
        if self.is_large_image:
            self._init_large_image_info()

    def _init_large_image_info(self):
        """自动适配：字典/路径 读取地理信息"""
        if self.is_dict_input:
            # 从crop字典直接拿地理信息（极速）
            self.height = self.pre_data["height"]
            self.width = self.pre_data["width"]
            self.geo_transform = self.pre_data["transform"]
            self.proj_wkt = self.pre_data["crs"]
        else:
            # 原路径模式：GDAL读取
            dataset = gdal.Open(self.pre_image_path, gdal.GA_ReadOnly)
            self.height = dataset.RasterYSize
            self.width = dataset.RasterXSize
            self.geo_transform = dataset.GetGeoTransform()
            self.proj_wkt = dataset.GetProjection()
            dataset = None

        # 计算分块行列（逻辑不变）
        self.rows = (self.height - self.patch_size + self.stride - 1) // self.stride + 1
        self.cols = (self.width - self.patch_size + self.stride - 1) // self.stride + 1
        self.length = self.rows * self.cols

    def __len__(self):
        return self.length if self.is_large_image else len(self.data_list)

    def __getitem__(self, index):
        return self._get_large_image_patch(index)

    def _get_large_image_patch(self, index):
    # 🔥 【和推理脚本完全一致的坐标计算】
        y = index // self.cols
        x = index % self.cols
        
        # 严格用 stride 计算，和你推理代码一模一样！
        y_start = y * self.stride
        x_start = x * self.stride
        y_end = min(y_start + self.patch_size, self.height)
        x_end = min(x_start + self.patch_size, self.width)
    
        if self.is_dict_input:
            # 维度修复（保留你成功的代码）
            pre_data = self.pre_data["image"]
            post_data = self.post_data["image"]
            
            # 🔥 【切片逻辑和推理完全对齐】
            pre_patch = pre_data[:3, y_start:y_end, x_start:x_end]
            post_patch = post_data[:3, y_start:y_end, x_start:x_end]
    
            # 地理变换
            geo_transform_patch = list(self.geo_transform)
            geo_transform_patch[0] += x_start * geo_transform_patch[1]
            geo_transform_patch[3] += y_start * geo_transform_patch[5]
    
        # 归一化
        pre_patch, post_patch, loc_label, clf_label = self.__transforms(False, pre_patch, post_patch, np.zeros((self.patch_size, self.patch_size)), 0)
    
        # 转张量
        pre_img = torch.from_numpy(pre_patch).float()
        post_img = torch.from_numpy(post_patch).float()
    
        # ✅ 完全保留你原来的返回值，一个都不多加！
        return (
            pre_img,
            post_img,
            torch.tensor(0, dtype=torch.long),
            torch.tensor(0, dtype=torch.long),
            "",
            geo_transform_patch,
            self.proj_wkt,
            self.rows,
            self.cols
        )



    def __transforms(self, aug, pre_img, post_img, loc_label, clf_label):
        pre_img = pre_img.transpose(1, 2, 0)  # [3,h,w] → [h,w,3]
        pre_img = imutils.normalize_img(pre_img)
        pre_img = pre_img.transpose(2, 0, 1)  # [h,w,3] → [3,h,w]
        
        # 处理后图
        post_img = post_img.transpose(1, 2, 0)
        post_img = imutils.normalize_img(post_img)
        post_img = post_img.transpose(2, 0, 1)
        return pre_img, post_img, loc_label, clf_label

    def save_final_results(self, final_output_clf, output_dir):
        if not self.is_large_image:
            return
        base_name = "memory_infer_result"
        clf_output_path = os.path.join(os.path.dirname(output_dir), f'{base_name}_clf.tif')
        save_geotiff(final_output_clf, clf_output_path, self.geo_transform, self.proj_wkt, no_data_value=0)
        tif_xml_single(clf_output_path, output_dir)
        os.remove(clf_output_path)





class DamageDataset_train(Dataset):
    def __init__(self, data_dir_list, crop_size, max_iters=None, type='train', data_loader=tif_loader_window,
                 is_large_image=False, patch_size=512, overlap=64, pure_inference=False):
        """
        纯推理模式数据集类
        :param pure_inference: 是否纯推理模式（不加载标签）
        """
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.is_large_image = is_large_image
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.crop_size = crop_size
        self.pure_inference = pure_inference  # 纯推理模式标志
        
        self.sample_dirs = data_dir_list
        
        # if self.pre_image_path:
        #     base_name = os.path.splitext(os.path.basename(self.pre_image_path))[0]
        #     self.data_list = [base_name]
        # else:
        #     self.data_list = []
        
        # if max_iters is not None:
        #     self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
        #     self.data_list = self.data_list[0:max_iters]
        
        # 大图模式初始化
        if self.is_large_image:
            self._init_large_image_info()
    
    def _init_large_image_info(self):
        """为大图模式初始化分块信息"""
        # pre_path = os.path.join(self.dataset_path, 'image/pre', self.data_list[0] + '.tif')
        pre_path = self.pre_image_path
        
        # 使用GDAL打开文件获取元数据
        dataset = gdal.Open(pre_path, gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError(f"无法打开文件: {pre_path}")
        
        self.height = dataset.RasterYSize
        self.width = dataset.RasterXSize
        
        # 获取地理变换和投影信息
        self.geo_transform = dataset.GetGeoTransform()
        self.proj_wkt = dataset.GetProjection()
        
        dataset = None  # 立即关闭
        
        # 计算网格行列数
        self.rows = (self.height - self.patch_size + self.stride - 1) // self.stride + 1
        self.cols = (self.width - self.patch_size + self.stride - 1) // self.stride + 1
        self.length = self.rows * self.cols
    
    def __len__(self):
        if self.is_large_image:
            return self.length  # 返回分块数量
        return len(self.sample_dirs)  # 返回小图数量
    
    def __getitem__(self, index):
        if self.is_large_image:
            return self._get_large_image_patch(index)
        else:
            return self._get_small_image(index)
        
    
    def _get_small_image(self, index):
        """获取小图 - 纯推理模式"""
        
        sample_group_dir = self.sample_dirs[index]
        
        base_img_dir = os.path.join(sample_group_dir, "base_image")
        
        comp_img_dir = os.path.join(sample_group_dir, "compare_image")
        
        label_dir = os.path.join(sample_group_dir, "label")
        
        
        try:
            # 找到 base_image 下的文件 (忽略隐藏文件)
            base_img_files = [f for f in os.listdir(base_img_dir) if not f.startswith('.')]
            if not base_img_files: raise FileNotFoundError(f"在 {base_img_dir} 中找不到图像文件")
            base_img_path = os.path.join(base_img_dir, base_img_files[0])

            # 找到 compare_image 下的文件
            comp_img_files = [f for f in os.listdir(comp_img_dir) if not f.startswith('.')]
            if not comp_img_files: raise FileNotFoundError(f"在 {comp_img_dir} 中找不到图像文件")
            comp_img_path = os.path.join(comp_img_dir, comp_img_files[0])
            
            # 找到 label 下的文件 (假设是 .tif 格式的 CLF 标签)
            label_files = [f for f in os.listdir(label_dir) if not f.startswith('.') and f.lower().endswith('.tif')]
            if not label_files: raise FileNotFoundError(f"在 {label_dir} 中找不到标签文件 (.tif)")
            label_path = os.path.join(label_dir, label_files[0])

        except Exception as e:
            print(f"[ERROR] 加载样本时出错 {sample_group_dir}: {e}", file=sys.stderr)
            # 返回 None 或其他错误指示，让 DataLoader 跳过这个样本
            # (需要DataLoader配置 collate_fn 来处理 None)
            # 或者，更简单地，返回一个无效的数据，然后在训练循环中跳过
            dummy_img = np.zeros((3, self.crop_size, self.crop_size), dtype=np.float32)
            dummy_label = np.zeros((self.crop_size, self.crop_size), dtype=np.int64) + 255 # 填充 ignore_index
            return dummy_img, dummy_img, dummy_label, sample_group_dir
        
        
        
        pre_img, geo_transform, proj_wkt = self.loader(base_img_path)
        
        post_img, _, _ = self.loader(comp_img_path)

        self.geo_transform = geo_transform
        self.proj_wkt = proj_wkt
                    
        if os.path.exists(label_path):
            clf_label, _, _ = self.loader(label_path)
        else:
            clf_label = np.zeros((pre_img.shape[0], pre_img.shape[1]), dtype=np.uint8)
        is_training = (self.type == 'train')
        # 应用预处理
        pre_img, post_img, clf_label = self.__transforms(
            is_training, pre_img, post_img, clf_label)
        
        # _,clf_label = airport(clf_label,clf_label)
        # _,clf_label = building(clf_label,clf_label)
        # clf_label[clf_label==44] = 2
        
        return pre_img, post_img, clf_label, sample_group_dir, geo_transform, proj_wkt
    
    def __transforms(self, aug, pre_img, post_img, clf_label):
        """预处理方法 - 纯推理模式下不做增强"""
        # print(f"DEBUG: __transforms called with aug={aug}, pure_inference={self.pure_inference}", file=sys.stderr, flush=True)
        
        if pre_img.ndim == 2:
            # 使用 np.stack 将单通道堆叠 3 次，创建 (H, W, 3) 数组
            pre_img = np.stack((pre_img,) * 3, axis=-1)
        elif pre_img.ndim == 3 and pre_img.shape[-1] == 1: # 如果是 (H, W, 1)
             pre_img = np.repeat(pre_img, 3, axis=-1) # 重复通道 -> (H, W, 3)

        # 对 post_img 做同样处理
        if post_img.ndim == 2:
            post_img = np.stack((post_img,) * 3, axis=-1)
        elif post_img.ndim == 3 and post_img.shape[-1] == 1:
             post_img = np.repeat(post_img, 3, axis=-1)
        
        if aug and not self.pure_inference:
            # 训练模式下的数据增强
            pre_img, post_img, clf_label = imutils.random_crop_bda(pre_img, post_img, clf_label, self.crop_size)
            pre_img, post_img, clf_label = imutils.random_fliplr_bda(pre_img, post_img, clf_label)
            pre_img, post_img, clf_label = imutils.random_flipud_bda(pre_img, post_img, clf_label)
            pre_img, post_img, clf_label = imutils.random_rot_bda(pre_img, post_img, clf_label)
        
        # 标准化
        pre_img = imutils.normalize_img(pre_img)
        pre_img = np.transpose(pre_img, (2, 0, 1))
        
        post_img = imutils.normalize_img(post_img)
        post_img = np.transpose(post_img, (2, 0, 1))
        
        return pre_img, post_img, clf_label
    
    def save_final_results(self, final_output_clf, output_dir,base_name):
        
        # 保存分类结果
        clf_output_path = os.path.join(output_dir, f'{base_name}_clf.tif')
        
        save_geotiff(final_output_clf, clf_output_path, self.geo_transform, self.proj_wkt, no_data_value=0)
    
        xml_path = os.path.join(output_dir, f'{base_name}.xml')
        
        tif_xml_single(clf_output_path, xml_path)
        
        os.remove(clf_output_path)
    
    


def make_data_loader(args, **kwargs):
    # 检查是否为大图模式
    is_large_image = getattr(args, 'large_image_mode', False) and len(args.train_data_name_list) == 1
    
    if 'test' in args.dataset:
        # 创建数据集，添加大图模式参数
        dataset = DamageDataset_train(
            args.train_data_dir_list, 
            args.crop_size, 
            args.max_iters, 
            args.type,
            is_large_image=is_large_image,
            patch_size=getattr(args, 'patch_size', 512),
            overlap=getattr(args, 'overlap', 64)
        )
        
        # 对于大图模式，可能需要调整batch_size
        batch_size = args.batch_size
        if is_large_image:
            # 大图模式下可能需要更大的batch_size以提高效率
            batch_size = min(args.batch_size * 2, 16)  # 最大不超过16
        
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=args.shuffle, 
            num_workers=0,
            drop_last=False
        )
        return data_loader
    
    else:
        raise NotImplementedError


def save_geotiff(output_array: np.ndarray, output_path: str, geo_transform, proj_wkt, no_data_value=None):
    """
    将 NumPy 数组保存为 GeoTIFF 文件，并嵌入地理坐标信息。

    参数:
        output_array (np.ndarray): 要保存的 NumPy 数组。预期形状为 (H, W) 或 (H, W, C)。
        output_path (str): 输出 GeoTIFF 文件的完整路径。
        geo_transform: 图像的地理仿射变换参数（来自 GDAL GetGeoTransform）。
        proj_wkt (str): 图像的投影信息（来自 GDAL GetProjection，WKT 格式）。
        no_data_value (any, optional): 要为 GeoTIFF 波段设置的 NoData 值。
    """
    driver = gdal.GetDriverByName('GTiff')
    
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 根据 NumPy 数组的数据类型确定 GDAL 数据类型
    if output_array.dtype == np.uint8:
        gdal_dtype = gdal.GDT_Byte
    elif output_array.dtype == np.uint16:
        gdal_dtype = gdal.GDT_UInt16
    elif output_array.dtype == np.int16:
        gdal_dtype = gdal.GDT_Int16
    elif output_array.dtype == np.float32:
        gdal_dtype = gdal.GDT_Float32
    elif output_array.dtype == np.float64:
        gdal_dtype = gdal.GDT_Float64
    else:
        # 对于未直接匹配的类型，回退到 Byte (uint8) 转换
        # print(f"警告: 数组数据类型 {output_array.dtype} 未直接匹配 GDAL 类型，将尝试转换为 Byte。")
        output_array = output_array.astype(np.uint8)
        gdal_dtype = gdal.GDT_Byte

    # 确定图像尺寸和波段数量
    if output_array.ndim == 2:
        rows, cols = output_array.shape
        num_bands = 1
    elif output_array.ndim == 3:
        # 假设多波段图像的形状是 (H, W, C)
        rows, cols, num_bands = output_array.shape
    else:
        raise ValueError("输出数组维度不正确，应为 2D (H,W) 或 3D (H,W,C)。")

    # 创建新的 GeoTIFF 文件
    out_dataset = driver.Create(output_path, cols, rows, num_bands, gdal_dtype)

    if out_dataset is None:
        print(f"错误: 无法创建输出 GeoTIFF 文件: {output_path}")
        return

    # 设置地理变换和投影信息
    out_dataset.SetGeoTransform(geo_transform)
    out_dataset.SetProjection(proj_wkt)

    # 写入数据到 GeoTIFF
    if num_bands == 1:
        out_dataset.GetRasterBand(1).WriteArray(output_array)
        if no_data_value is not None:
            out_dataset.GetRasterBand(1).SetNoDataValue(no_data_value)
    else:
        # 如果是多波段数组，则逐波段写入
        for i in range(num_bands):
            out_dataset.GetRasterBand(i + 1).WriteArray(output_array[:, :, i])
            if no_data_value is not None:
                out_dataset.GetRasterBand(i + 1).SetNoDataValue(no_data_value)

    # 刷新缓存并关闭数据集以保存文件
    out_dataset.FlushCache()
    out_dataset = None # 显式关闭数据集



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SECOND DataLoader Test")
    parser.add_argument('--dataset', type=str, default='test')
    parser.add_argument('--max_iters', type=int, default=None)
    parser.add_argument('--type', type=str, default='test')
    parser.add_argument('--train_dataset_path', type=str, default=r'E:/MambaCD/data_test')
    parser.add_argument('--train_data_list_path', type=str, default=r'E:/MambaCD/data_test/test.txt')
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_name_list', type=list)
    parser.add_argument('--crop_size', type=int, default=512)
    
    args = parser.parse_args()

    print("完整路径:", os.path.abspath(args.train_data_list_path))  # 验证实际访问路径
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        train_data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = train_data_name_list
    # train_data_loader = make_data_loader(args)
    # for i, data in enumerate(train_data_loader):
    #     pre_img, post_img, labels, _, _ = data
    #     pre_data, post_data = Variable(pre_img), Variable(post_img)
    #     labels = Variable(labels)
    #     print(i, "个inputs", pre_data.data.size(), "labels", labels.data.size())
    # dataset = DamageAssessmentDatset(args.train_dataset_path, args.train_data_name_list, 256, None, 'test')
    dataset = DamageDataset(args.train_dataset_path, args.train_data_name_list, 256, None, 'test')
    val_data_loader = DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False)
    print("验证集样本数:", len(val_data_loader.dataset))
    total_samples = 0
    loc_label_counter = Counter()
    clf_label_counter = Counter()
    val = tqdm(val_data_loader,desc="counting")
    
    for i, data in enumerate(val):
        pre_img, post_img, loc_labels, clf_labels, _ = data
        batch_size = pre_img.size(0)
        total_samples += batch_size
    
        # === 统计分类标签（如：0,1,2,3,4）=== 
        if clf_labels.ndimension() == 3:  # 如果是 [1, 1024, 1024] 的形状
            # 展平 clf_labels 为一维
            unique, counts = torch.unique(clf_labels, return_counts=True)
            clf_label_counter.update(dict(zip(unique.tolist(), counts.tolist())))
        
        # === 统计定位标签（像素级标签，如分割图）=== 
        if loc_labels.ndimension() >= 3:
            unique, counts = torch.unique(loc_labels, return_counts=True)
            loc_label_counter.update(dict(zip(unique.tolist(), counts.tolist())))
        else:
            loc_label_counter.update(loc_labels.tolist())
    
    print(f"\n总样本数：{total_samples}")
    
    print("\n 分类标签（clf_labels）分布：")
    for k in sorted(clf_label_counter.keys()):
        print(f"  类别 {k}: {clf_label_counter[k]}")
    
    print("\n 定位标签（loc_labels）分布：")
    for k in sorted(loc_label_counter.keys()):
        print(f"  类别 {k}: {loc_label_counter[k]}")
