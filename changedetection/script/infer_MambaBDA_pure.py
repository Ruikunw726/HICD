import sys
from pathlib import Path
import os
import json
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



PROJECT_ROOT = FILE.parents[3] 

os.chdir(PROJECT_ROOT)

import argparse
import time
from PIL import Image
import numpy as np

from MambaCD.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from MambaCD.changedetection.datasets.make_data_loader_pure import make_data_loader,DamageDataset_infer
from MambaCD.changedetection.utils_func.metrics import Evaluator
from MambaCD.changedetection.models.STMambaBDA import STMambaBDA

import imageio
import numpy as np
import seaborn as sns
from osgeo import gdal, osr
gdal.UseExceptions()

os.environ['PROJ_LIB']='/usr/share/proj'
os.environ['GDAL_DATA']='/usr/share/gdal'
os.environ['GTIFF_SRS_SOURCE']='EPSG'


ori_label_value_dict = {
    'background': (0, 0, 0),      
    'no_damage': (70, 181, 121),     
    'minor_damage': (167, 187, 27),   
    'majoy_damage': (236, 119, 31),  
    'destroyed': (212, 59, 59)        
}

target_label_value_dict = {
    'background': 0,
    'no_damage': 1,
    'minor_damage': 2,
    'majoy_damage': 3,
    'destroyed':4
}

loc_target_label_value_dict = {
    'background': 0,         # 黑色 - 背景
    'urban': 1,            # 深红 - 城市区域
    'agriculture': 2,       # 深绿 - 农业用地
    'rangeland': 3,      # 橄榄绿 - 牧场
    'forest': 4,           # 深蓝 - 森林
    'water': 5,           # 紫色 - 水域
    'barren': 6,          # 青色 - 荒地
    'unknown': 7      # 灰色 - 未知区域
}

# LOC标签的7类颜色映射（位置识别）
loc_ori_label_value_dict = {
    'background': (0, 0, 0),         # 黑色 - 背景
    'urban': (128, 0, 0),            # 深红 - 城市区域
    'agriculture': (0, 128, 0),       # 深绿 - 农业用地
    'rangeland': (128, 128, 0),      # 橄榄绿 - 牧场
    'forest': (0, 0, 128),           # 深蓝 - 森林
    'water': (128, 0, 128),           # 紫色 - 水域
    'barren': (0, 128, 128),          # 青色 - 荒地
    'unknown': (192, 192, 192)       # 灰色 - 未知区域
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

def loc_map_labels_to_colors(labels, ori_label_value_dict, target_label_value_dict):
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
        self.evaluator_clf = Evaluator(num_class=5)
        self.total_evaluator_loc = Evaluator(num_class=2)
        self.total_evaluator_clf = Evaluator(num_class=5)

        self.deep_model = STMambaBDA(
            output_building=2, output_damage=5,
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

        # self.building_map_T1_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'classification_map_single_airport_test')
        # self.change_map_T2_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'damage_map_single_airport_test')
        
        # self.cls_map_T1_saved_path = os.path.join(os.path.dirname(self.args.result_saved_path), 'classification')
        self.change_map_T2_saved_path = os.path.dirname(self.args.result_saved_path)


        # if not os.path.exists(self.cls_map_T1_saved_path):
        #     os.makedirs(self.cls_map_T1_saved_path)
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

        # 创建数据集
        dataset = DamageDataset_infer(
            self.args.input_data,
            self.args.input_data2,
            256, None, 'test',
            is_large_image=getattr(self.args, 'large_image_mode', True),
            patch_size=getattr(self.args, 'patch_size', 512),
            overlap=getattr(self.args, 'overlap', 32),
            pure_inference=getattr(self.args,'pure_inference',True)
        )
        
        # 创建数据加载器
        val_data_loader = DataLoader(
            dataset, 
            batch_size=self.args.batch_size, 
            num_workers=0, 
            drop_last=False
        )
        
        torch.cuda.empty_cache()
        
        try:
            total_batches = len(val_data_loader)
            if total_batches == 0:
                print("[WARN] 数据加载器为空, 无法报告进度。")
                sys.stdout.flush()
        except Exception as e:
            print(f"[ERROR] 无法获取 val_data_loader 长度: {e}")
            sys.stdout.flush()
            total_batches = -1 # 设为-1, 避免后续计算
        
        # 大图模式：初始化结果数组
        if dataset.is_large_image:
            height = dataset.height
            width = dataset.width
            final_output_loc = np.zeros((height, width), dtype=np.uint8)
            final_output_clf = np.zeros((height, width), dtype=np.uint8)
            # count_matrix = np.zeros((height, width), dtype=np.uint8)
        
        last_percentage = 50  # 初始值和你一致
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                # 根据数据集模式加载数据
                if dataset.is_large_image:
                    pre_change_imgs, post_change_imgs, _, _, names, geo_trans, proj, rows, cols = data
                else:
                    pre_change_imgs, post_change_imgs, _, _, names, geo_trans, proj = data
                
                                            
                proj = proj[0]
                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                              
                # 1. 获取当前切片的真实大小
                _, _, h1, w1 = pre_change_imgs.shape
                _, _, h2, w2 = post_change_imgs.shape
                
                # 2. 取最小尺寸（无需 32 倍数）
                target_h = min(h1, h2)
                target_w = min(w1, w2)
                
                # 3. 中心裁剪
                def center_crop(img, th, tw):
                    _, _, h, w = img.shape
                    start_h = (h - th) // 2
                    start_w = (w - tw) // 2
                    return img[:, :, start_h:start_h+th, start_w:start_w+tw]
                
                pre_change_imgs = center_crop(pre_change_imgs, target_h, target_w)
                post_change_imgs = center_crop(post_change_imgs, target_h, target_w)
                
                # 4. 直接推理（尺寸相同即可）
                output_clf = self.deep_model(pre_change_imgs, post_change_imgs)
                
                # ========================================================
                # 【结果裁剪】把补上去的黑边裁掉，还原回 A 的原始尺寸
                # ========================================================
                # 注意：我们通常以 Pre (A) 的尺寸为基准输出
                if output_clf.shape[2] != h1 or output_clf.shape[3] != w1:
                    output_clf = output_clf[:, :, :h1, :w1]
                

                if total_batches > 0:
                    # 计算当前进度
                    percentage = 50 + int(((itera + 1) / total_batches) * 50)
                    
                    # 🔥 核心：只有进度变化了，才打印！
                    if percentage != last_percentage:
                        print(f"Progress:{percentage}%")
                        sys.stdout.flush()
                        last_percentage = percentage  # 更新记录

                # image1_name = names[0] + '_loc_visual.png'
                image2_name = names[0] + '_clf_visual.png'
                if dataset.is_large_image:
                    output_clf = output_clf.data.cpu().numpy()
                    output_clf = np.argmax(output_clf, axis=1)
                    
                    # 🔥 核心修复：获取总块行列（固定值，不是遍历）
                    total_patch_rows = dataset.rows
                    total_patch_cols = dataset.cols
                    
                    # 🔥 核心修复：遍历批次内的每个patch，用itera计算索引
                    for i in range(output_clf.shape[0]):
                        # ✅ 正确：当前patch的全局索引 = 批次起始索引 + i
                        patch_idx = itera * self.args.batch_size + i
                        
                        # ✅ 正确：计算当前patch的行、列
                        row = patch_idx // total_patch_cols
                        col = patch_idx % total_patch_cols
                        
                        # ✅ 坐标计算（和数据集完全一致，不回退）
                        y = row * dataset.stride + (h1 - target_h) // 2  # 加上裁剪偏移
                        x = col * dataset.stride + (w1 - target_w) // 2
                        
                        y = max(0, y)
                        x = max(0, x)
                        y_end = min(y + dataset.patch_size, height)
                        x_end = min(x + dataset.patch_size, width)
                        
                        # 填充结果（完全不变）
                        patch_h = y_end - y
                        patch_w = x_end - x
                        clf_patch = output_clf[i, :patch_h, :patch_w]
                        final_output_clf[y:y_end, x:x_end] = clf_patch
                else:
                    # 小图模式：直接保存结果
                    
                    output_clf = output_clf.data.cpu().numpy()
                    output_clf = np.argmax(output_clf, axis=1)
                    
                    # 保存小图结果
                    image_name = names[0] + '.tif'
                    
                    # 保存原始结果
                    # loc_output_path = os.path.join(self.args.result_saved_path, 'loc', image_name)
                    clf_output_path = os.path.join(self.args.result_saved_path, 'clf', image_name)
                    
                    # 确保目录存在
                    # os.makedirs(os.path.dirname(loc_output_path), exist_ok=True)
                    os.makedirs(os.path.dirname(clf_output_path), exist_ok=True)
                    
                    # 保存GeoTIFF
                    save_geotiff(output_clf, clf_output_path, geo_trans[0], proj, no_data_value=0)
                    
                    # save_geotiff(output_loc, loc_output_path, geo_trans[0], proj, no_data_value=0)
                    
                    # 保存可视化结果

                    
                    clf_color = map_labels_to_colors(output_clf, 
                                                  ori_label_value_dict=ori_label_value_dict, 
                                                  target_label_value_dict=target_label_value_dict)
                    
                    imageio.imwrite(os.path.join(self.args.result_saved_path, 'clf_visual', image2_name), clf_color.astype(np.uint8))
        
        # 大图模式：保存完整结果
        if dataset.is_large_image:
             
            dataset.save_final_results(final_output_clf, self.args.result_saved_path)
                    
            final_output_clf1 = map_labels_to_colors(np.squeeze(final_output_clf), ori_label_value_dict=ori_label_value_dict, target_label_value_dict=target_label_value_dict)
           
            # final_output_clf1[final_output_loc == 0] = 0
            
            # dataset.save_final_results(final_output_loc, final_output_clf1, self.args.result_saved_path)
            
            # final_output_loc1 = loc_map_labels_to_colors(final_output_loc, ori_label_value_dict=loc_ori_label_value_dict, 
                                                  # target_label_value_dict=loc_target_label_value_dict)
            # imageio.imwrite(os.path.join(os.path.dirname(self.args.result_saved_path), image1_name), final_output_loc1.astype(np.uint8))
            imageio.imwrite(os.path.join(os.path.dirname(self.args.result_saved_path), image2_name), final_output_clf1.astype(np.uint8))
                
            


def main():
    inference_start = time.time()
    
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    
    parser = argparse.ArgumentParser(description="Inference on test dataset")
    parser.add_argument('--cfg', type=str, default=r'./MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='test')
    parser.add_argument('--type', type=str, default='test')
    parser.add_argument('--input_data', type=str, default=r'/home/user/桌面/A/12-1.tif')
    parser.add_argument('--input_data2', type=str, default=r'/home/user/桌面/B/12-1.tif')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--model_type', type=str, default='MambaBDA_Tiny')
    parser.add_argument('--result_saved_path', type=str, default=r'/home/user/桌面/results/result.xml')
    parser.add_argument('--large_image_mode', type=bool, default=True)
    
    parser.add_argument('--pure_inference', type=bool, default=True)
    parser.add_argument('--slice', type=str, default=None)
    
    parser.add_argument('--resume', type=str,
        # default=r'E:/baidu_download/Changemamba/MambaCD/changedetection/saved_models/xBD/MambaBDA_Small_1743787146.0769622/37500_model.pth'
        default='/home/user/桌面/399.pth'
    )
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()

    # with open(args.test_data_list_path, "r") as f:
    #     # data_name_list = f.read()
    #     test_data_name_list = [data_name.strip() for data_name in f]
    # args.test_data_name_list = test_data_name_list
    
    
    ############ 配准流程
        
    base_dir = os.path.dirname(args.result_saved_path)
    os.makedirs(base_dir, exist_ok=True)
    
    # 这里的名字可以起个带有 temp 标记的
    temp_registered_filename = f"temp_registered.tif"
    temp_registered_path = os.path.join(base_dir, temp_registered_filename)
    
    # 标记是否成功生成了临时文件，用于后面判断是否需要删除
    has_temp_file = False

    try:
        crop_points_test = [
            [51.290000, 25.100000],  # 左上（经度，纬度）
            [51.335000, 25.100000],  # 右上
            [51.290000, 25.090000],  # 左下
            [51.335000, 25.090000]   # 右下
        ]  
        from MambaCD.func.crop import crop2
        if args.slice is not None:
            crop_points = json.loads(args.slice)
        else:
            crop_points = None
        print("image crop started")
        print("Progress:5%")
        has_temp_cropfile = False
        crop_pre,crop_post = crop2(
            args.input_data,
            args.input_data2,
            crop_points
            )
        has_temp_cropfile = True
        # print(f"Crop后: crop_pre type={type(crop_pre)}, crop_post type={type(crop_post)}")
        args.input_data = crop_pre
        args.input_data2 = crop_post
        # aligned_paths = [pre_aligned_path,post_aligned_path]
        print("image crop success")
    
    except Exception as e:
        print(f"crop failed,using aligned original images:{e}")


    try:
        # 2. 执行配准
        from MambaCD.func.MapGlue_main.test import register_images
        
        temp_pre,temp_post=register_images(
            img1_path=args.input_data2,
            img2_path=args.input_data,
            output_path=temp_registered_path, # 保存到临时路径
            num_keypoints=512,
            grid_step_deg=0.003,
            model_path='./MambaCD/func/MapGlue_main/weights/fastmapglue_model.pt',
            generate_chessboard=False
        )     
        # 3. 替换输入路径
        args.input_data = temp_pre
        args.input_data2 = temp_post
        
        # print("配准返回的 temp_pre 类型：", type(temp_pre))
        # print("配准返回的 temp_pre 值：", temp_pre)
        # print("配准返回的 temp_post 类型：", type(temp_post))
        # print("配准返回的 temp_post 值：", temp_post)
        has_temp_file = True
        
    except Exception as e:
        print(f"regis failed: {e}")
        print("using oringinal images to continue...")
        
    # try:
    #     crop_points_test = [
    #         [51.290000, 25.100000],  # 左上（经度，纬度）
    #         [51.335000, 25.100000],  # 右上
    #         [51.290000, 25.090000],  # 左下
    #         [51.335000, 25.090000]   # 右下
    #     ]  
    #     from MambaCD.func.crop import crop2
    #     if args.slice is not None:
    #         crop_points = json.loads(args.slice)
    #     else:
    #         crop_points = None
    #     print("image crop started")
    #     print("Progress:40%")
    #     has_temp_cropfile = False
    #     crop_pre,crop_post = crop2(
    #         args.input_data,
    #         args.input_data2,
    #         crop_points
    #         )
    #     has_temp_cropfile = True
        
    #     args.input_data = crop_pre
    #     args.input_data2 = crop_post
    #     # aligned_paths = [pre_aligned_path,post_aligned_path]
    #     print("image crop success")
    #     print("Progress:45%")
    
    # except Exception as e:
    #     print(f"crop failed,using aligned original images:{e}")
    # =========================================================================
    # [Stage 1] 推理 (包裹在 try...finally 中)
    # =========================================================================
    # 正常执行推理
    trainer = Trainer(args)
    trainer.infer()
        

    # finally:
    #     # =====================================================================
    #     # [Stage 2] 自动清理垃圾
    #     # =====================================================================
    #     # 只有当我们确实生成了临时文件，且该文件还在硬盘上时，才删除
    #     if has_temp_file and os.path.exists(temp_registered_path):
    #         try:
    #             time.sleep(0.5) 
    #             os.remove(temp_registered_path)
    #         except Exception as e:
    #             print(f"warning temp registered filed cannot be deleted: {e}")
    #             print(f"please delete manually:{temp_registered_path}")
    #     if has_temp_cropfile:
    #         for p in aligned_paths:
    #             if os.path.exists(p):
    #                 try:
    #                     os.remove(p)
    #                 except Exception as e:
    #                     print(f"delete crop files failed:{e}")
    #                     print(f"please delete manually:{p}")
    inference_end = time.time()
    print(f"complete，time cost: {inference_end - inference_start:.2f} s")
    ########## 配准流程
        

    # trainer = Trainer(args)
    # trainer.infer()
    
    # inference_end = time.time()
    # inference_time = inference_end - inference_start
    # print(f"推理完成，总耗时: {inference_time:.2f} 秒")


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

if __name__ == "__main__":
    main()



