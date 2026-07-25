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

import argparse
import os
import time

import numpy as np

from MambaCD.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from MambaCD.changedetection.datasets.make_data_loader import make_data_loader,DamageDataset
from MambaCD.changedetection.utils_func.metrics import Evaluator
from MambaCD.changedetection.models.STMambaBDA import STMambaBDA
from MambaCD.func.tif_to_json import tif_json
import imageio
import numpy as np
import seaborn as sns
from osgeo import gdal, osr
gdal.UseExceptions()

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

def map_labels_to_colors(labels, ori_label_value_dict, target_label_value_dict):
    # Reverse the target_label_value_dict to get a mapping from target labels to original labels
    target_to_ori = {v: k for k, v in target_label_value_dict.items()}
    
    # Initialize an empty 3D array for the color-mapped labels
    H, W = labels.shape
    color_mapped_labels = np.zeros((H, W, 3), dtype=np.uint8)
    
    for target_label, ori_label in target_to_ori.items():
        # Find where the label matches the current target label
        mask = labels == target_label
        
        # Map these locations to the corresponding color value
        color_mapped_labels[mask] = ori_label_value_dict[ori_label]
    
    return color_mapped_labels


class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=3)
        self.total_evaluator_loc = Evaluator(num_class=2)
        self.total_evaluator_clf = Evaluator(num_class=3)

        self.deep_model = STMambaBDA(
            output_building=2, output_damage=3,
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        ) 
        self.deep_model = self.deep_model.cuda()
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        # self.building_map_T1_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'building_localization_map')
        self.change_map_T2_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'damage_classification_map')

        # if not os.path.exists(self.building_map_T1_saved_path):
            # os.makedirs(self.building_map_T1_saved_path)
        if not os.path.exists(self.change_map_T2_saved_path):
            os.makedirs(self.change_map_T2_saved_path)


        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.deep_model.eval()


    def infer(self):
        torch.cuda.empty_cache()
        # dataset = DamageAssessmentDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        dataset = DamageDataset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False)
        print("验证集样本数:", len(val_data_loader.dataset))
        torch.cuda.empty_cache()
        self.total_evaluator_loc.reset()
        self.total_evaluator_clf.reset()          
        # vbar = tqdm(val_data_loader, ncols=50)
        with torch.no_grad():
            for itera, data in enumerate(tqdm(val_data_loader)):
                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, names,geo_trans,proj = data
                proj = proj[0]
                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()


                output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)

                output_loc = output_loc.data.cpu().numpy()
                output_loc = np.argmax(output_loc, axis=1)
                labels_loc = labels_loc.cpu().numpy()

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()

                self.total_evaluator_loc.add_batch(labels_loc, output_loc)
                
                output_clf_eval = output_clf[labels_loc > 0]
                labels_clf_eval = labels_clf[labels_loc > 0]
                
                self.total_evaluator_clf.add_batch(labels_clf_eval, output_clf_eval)

                # loc_unique, loc_counts = torch.unique(torch.tensor(output_loc), return_counts=True)
                # clf_unique, clf_counts = torch.unique(torch.tensor(output_clf), return_counts=True)
        
                # # Output distributions
                # print(f"loc distribution (iter {itera}):")
                # for label, count in zip(loc_unique.numpy(), loc_counts.numpy()):
                #     print(f"  Label {label}: {count} pixels")
        
                # print(f"clf distribution (iter {itera}):")
                # for label, count in zip(clf_unique.numpy(), clf_counts.numpy()):
                #     print(f"  Label {label}: {count} pixels")

                image_name = names[0] + '.tif'
                

                output_loc = np.squeeze(output_loc)
                output_loc[output_loc > 0] = 255

                output_clf1 = map_labels_to_colors(np.squeeze(output_clf), ori_label_value_dict=ori_label_value_dict, target_label_value_dict=target_label_value_dict)

                output_clf1[output_loc == 0] = 0
                
                output_clf = np.squeeze(output_clf)
                output_clf[output_loc == 0] = 0
                
                # loc_output_path = os.path.join(self.building_map_T1_saved_path, image_name)
                # save_geotiff(output_loc, loc_output_path,geo_trans, proj,no_data_value=0) # 通常背景设为NoData
                
                clf_output_path = os.path.join(self.change_map_T2_saved_path, image_name)
                save_geotiff(output_clf, clf_output_path,geo_trans, proj,no_data_value=0) # 通常背景设为NoData
                
                image1_name = names[0] + '_visual.png'
                
                output_json_name = names[0] + '.json'
                # imageio.imwrite(os.path.join(self.building_map_T1_saved_path, image1_name), output_loc.astype(np.uint8))
                imageio.imwrite(os.path.join(self.change_map_T2_saved_path, image1_name), output_clf1.astype(np.uint8))
                
                tif_json(self.change_map_T2_saved_path, self.change_map_T2_saved_path)

        loc_f1_score = self.total_evaluator_loc.Pixel_F1_score()
        damage_f1_score,precision,recall = self.total_evaluator_clf.Damage_F1_socore()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        oaf1 = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1
        print(f'lofF1 is {loc_f1_score}, clfF1 is {harmonic_mean_f1}, oaF1 is {oaf1}, '
              f'sub class F1 score is {damage_f1_score}')
        print(f'precision is {precision}, recall is {recall}')
        
        


def main():
    parser = argparse.ArgumentParser(description="Inference on test dataset")
    parser.add_argument('--cfg', type=str, default='./MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='test')
    parser.add_argument('--type', type=str, default='test')
    parser.add_argument('--test_dataset_path', type=str, default='./MambaCD/test')
    parser.add_argument('--test_data_list_path', type=str, default='./MambaCD/test/test.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--model_type', type=str, default='MambaBDA')
    parser.add_argument('--result_saved_path', type=str, default='../results')

    parser.add_argument('--resume', type=str,
        # default=r'E:/baidu_download/Changemamba/MambaCD/changedetection/saved_models/xBD/MambaBDA_Small_1743787146.0769622/37500_model.pth'
        default='./MambaCD/changedetection/saved_models/test/data_test_model.pth'
    )
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.infer()


def save_geotiff(output_array: np.ndarray, output_path: str, geo_transform, proj_wkt, no_data_value=None):
    """
    将 NumPy 数组保存为 GeoTIFF 文件，并嵌入地理坐标信息。
    """
    driver = gdal.GetDriverByName('GTiff')

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

if __name__ == "__main__":
    main()



