import argparse
import os
import sys
sys.path.append('/root/autodl-tmp')

import imageio
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm
import torch
from osgeo import gdal
gdal.UseExceptions()

import HICD.changedetection.datasets.imutils as imutils


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img

def img_loader_3(path):
    img = np.array(imageio.imread(path), np.float32)
    img = img[:,:,:3]
    return img

def tif_loader(path):
    """
    使用 GDAL 读取 GeoTIFF 图像，并只取前三个波段，保留地理坐标信息。
    """
    dataset = gdal.Open(path, gdal.GA_ReadOnly)

    if dataset is None:
        print(f"无法打开文件: {path}")
        return None, None, None

    # 获取图像的地理变换和投影信息
    geo_transform = dataset.GetGeoTransform()
    proj_wkt = dataset.GetProjection()

    # 读取所有波段数据
    # GDAL 默认读取为 (C, H, W) 顺序
    all_bands_img = dataset.ReadAsArray()

    # 关闭数据集，释放资源
    dataset = None

    # 检查维度。如果不是3维（单波段图像是2维），或者波段数不足，
    # 则需要特别处理，或者抛出错误，这里我们只取前3个波段
    if all_bands_img.ndim == 3:
        # 如果波段数大于3，则只取前三个波段
        if all_bands_img.shape[0] >= 3:
            img = all_bands_img[:3, :, :] # 截取前三个波段 (C, H, W)
            img = np.transpose(img, (1, 2, 0)) # 将波段轴移动到最后，变为 (H, W, C)
        else:
            # 如果波段数不足3，则警告并使用所有可用波段
            print(f"警告: 文件 {path} 的波段数不足 3 个（当前有 {all_bands_img.shape[0]} 个波段），将使用所有可用波段。")
            img = np.transpose(all_bands_img, (1, 2, 0)) # 变为 (H, W, C)
    else:
        # 如果是单波段图像（2维），直接赋值，不进行波段截取或转置
        img = all_bands_img # 形状是 (H, W)


    # 确保数据类型为 float32
    img = img.astype(np.float32)

    return img, geo_transform, proj_wkt
    


def one_hot_encoding(image, num_classes=8):
    # Create a one hot encoded tensor
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    # Move the channel axis to the front
    # one_hot = np.moveaxis(one_hot, -1, 0)

    return one_hot


# add
class DamageDataset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=tif_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        # 修改为读取tif的loader
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img,loc_label,clf_label):
        if aug:
            pre_img, post_img, loc_label, clf_label = imutils.random_crop_bda(pre_img, post_img, loc_label, clf_label, self.crop_size)
            pre_img, post_img, loc_label, clf_label = imutils.random_fliplr_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_flipud_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_rot_bda(pre_img, post_img, loc_label, clf_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, loc_label, clf_label

    def __getitem__(self, index):
        
        pre_path = os.path.join(self.dataset_path, 'image/pre', self.data_list[index] + '.tif')
        post_path = os.path.join(self.dataset_path, 'image/post', self.data_list[index] + '.tif')
        loc_label_path = os.path.join(self.dataset_path, 'label/loc', self.data_list[index]+ '.tif')
        clf_label_path = os.path.join(self.dataset_path, 'label/clf', self.data_list[index]+ '.tif')


        pre_img, geo_transform, proj_wkt = self.loader(pre_path)
        post_img,_,_ = self.loader(post_path)
        # loc_label = self.loader(loc_label_path)[:, :, 0]
        # clf_label = self.loader(clf_label_path)[:, :, 0]
        loc_label,_,_ = self.loader(loc_label_path)
        clf_label,_,_ = self.loader(clf_label_path)


        if 'train' in self.data_pro_type:
            pre_img, post_img, loc_label, clf_label = self.__transforms(
                True, pre_img, post_img, loc_label, clf_label)   

            
        else:
            pre_img, post_img, loc_label, clf_label = self.__transforms(
                False, pre_img, post_img, loc_label, clf_label)
            loc_label = np.asarray(loc_label)
            clf_label = np.asarray(clf_label)
        
        # 根据标签的字段值自行调整
        clf_label[clf_label == 150] = 2
        clf_label[clf_label == 255] = 1
        clf_label[clf_label == 0] = 255
        
        loc_label[loc_label > 1] = 1
        
        
        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx, geo_transform, proj_wkt

    def __len__(self):
        return len(self.data_list)


def make_data_loader(args, **kwargs):  # **kwargs could be omitted
    # add
    if 'test' in args.dataset:
        dataset = DamageDataset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=0,
                                 drop_last=False)
        return data_loader
    
    else:
        raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SECOND DataLoader Test")
    parser.add_argument('--dataset', type=str, default='test')
    parser.add_argument('--max_iters', type=int, default=None)
    parser.add_argument('--type', type=str, default='test')
    parser.add_argument('--train_dataset_path', type=str, default=r'E:/HICD/data_test')
    parser.add_argument('--train_data_list_path', type=str, default=r'E:/HICD/data_test/test.txt')
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
