import sys
from pathlib import Path
import json
import os

import warnings

# 忽略关于 torch.cuda.amp.custom_bwd 的 FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

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
import os
import time

import numpy as np

from MambaCD.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from MambaCD.changedetection.datasets.make_data_loader_pure import DamageDataset_train
from MambaCD.changedetection.utils_func.metrics import Evaluator
from MambaCD.changedetection.models.STMambaBDA import STMambaBDA

import MambaCD.changedetection.utils_func.lovasz_loss as L


class Trainer(object):
    def __init__(self, args):
        
        self.args = args
        config = get_config(args)

        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=5)

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
        self.model_save_path = self.args.model_param_path
        self.lr = args.learning_rate
        # self.epoch = args.max_iters // args.batch_size

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

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
            
            
            # state_dict = {k: v for k, v in checkpoint.items() 
            #     if "main_clf" not in k}
            self.deep_model.load_state_dict(state_dict,strict=False)
            
            
            # self.deep_model.load_state_dict(state_dict,strict=False)
        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
        
    def save(self,epoch_num):
        
        save_path = os.path.join(self.model_save_path, f'{epoch_num}.pth')
       
        torch.save(self.deep_model.state_dict(),save_path)


    def validation(self):
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        dataset = DamageDataset_train(self.args.test_data_dir_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False)
        torch.cuda.empty_cache()
        
        epoch_val_losses = [] # [新增] 用于计算平均验证损失
        
        total_samples = len(self.args.test_data_dir_list)
        
        # vbar = tqdm(val_data_loader, ncols=50)
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels_clf, sample_ids ,_,_ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_clf = labels_clf.cuda().long()
                 
                
                # input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
                output_clf = self.deep_model(pre_change_imgs, post_change_imgs)

                ce_loss_clf = F.cross_entropy(output_clf, labels_clf, ignore_index=255)
                
                lovasz_loss_clf = L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_clf, ignore=255)
                
                
                final_loss =  ce_loss_clf + 0.75 * lovasz_loss_clf
                
                epoch_val_losses.append(final_loss.item())                

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()
                
                # clf_unique, clf_counts = torch.unique(torch.tensor(labels_clf), return_counts=True)
                # for label, count in zip(clf_unique.numpy(), clf_counts.numpy()):
                #     print(f"\n  Label {label}: {count} pixels\n")
                # print("\n")
                
                
                # clf_unique, clf_counts = torch.unique(torch.tensor(labels_clf), return_counts=True)
                # for label, count in zip(clf_unique.cpu().detach().numpy(), clf_counts.cpu().detach().numpy()):
                #     print(f"\n  Label {label}: {count} pixels\n")    
                # print("\n")
                
                self.evaluator_clf.add_batch(labels_clf, output_clf)
                
                current_sample_id = sample_ids[0]
                comp_img_dir = os.path.join(current_sample_id, "compare_image")
                if os.path.isdir(comp_img_dir):
                        comp_files = [f for f in os.listdir(comp_img_dir) if not f.startswith('.') and f.lower().endswith(('.tif', '.tiff'))]
                        if comp_files:
                            comp_img_filename = comp_files[0]
                if comp_img_filename:
                        base_name = os.path.splitext(comp_img_filename)[0]
                else:
                    # 如果找不到后时相文件，则回退到使用目录名或生成备用名
                    base_name = os.path.basename(current_sample_id)
                    print(f"[Warning] Cannot find compare_image filename in {comp_img_dir}. Using directory name '{base_name}' as base_name.", file=sys.stderr)
                    if not base_name: base_name = f"sample_{itera}" # 备用名

                dataset.save_final_results(output_clf, self.model_save_path,base_name)
                
                percentage = int(((itera + 1) / total_samples) * 100)
                # 按照指定格式打印进度
                print(f"Progress:{percentage}%")
                # 立即刷新标准输出，确保外部进程能实时捕获
                sys.stdout.flush()
                
        # avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else 0.0
        # loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        # damage_f1_score,precision,recall = self.evaluator_clf.Damage_F1_socore()
        # harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        # oaf1 = harmonic_mean_f1
        
        
        # print(f'lofF1 is {loc_f1_score}, clfF1 is {harmonic_mean_f1}, oaF1 is {oaf1}, '
        #       f'sub class F1 score is {damage_f1_score}')
        # print(f'precision is {precision}, recall is {recall}')
        # return avg_val_loss, harmonic_mean_f1, precision, recall, damage_f1_score
        return 0


def main():
    parser = argparse.ArgumentParser(description="Training on test dataset")
    parser.add_argument('--cfg', type=str, default=r'./MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)

    parser.add_argument('--dataset', type=str, default='test')
    parser.add_argument('--type', type=str, default='train')
    
    parser.add_argument('--input_data', type=str, default=r'D:\docker环境_2\test\MambaCD\frame_building')
    
   
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--crop_size', type=int, default=256)
    # parser.add_argument('--train_data_name_list', type=list) 
    # parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=16000)
    parser.add_argument('--model_type', type=str, default='STMambaBDA')
    parser.add_argument('--model_param_path', type=str, default='D:/docker环境_2/MambaCD/changedetection/results')
    
    parser.add_argument('--training_log', type=str, default='D:/docker环境_2/MambaCD/changedetection/saved_models/training_log.txt')
    
    parser.add_argument('--pure_inference', type=bool, default=False)

    parser.add_argument('--resume', type=str,
        # default=r'E:/baidu_download/Changemamba/MambaCD/changedetection/saved_models/xBD/MambaBDA_Small_1743787146.0769622/37500_model.pth')
        # default=None)
        default=r'./MambaCD/changedetection/saved_models/43.pth')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)

    args = parser.parse_args()
    
    args.train_dataset_path = os.path.join(args.input_data, 'train') # 规范
    args.test_dataset_path = os.path.join(args.input_data, 'test')
    
    # args.test_dataset_path = args.input_data
    
    
    
    def scan_sample_dirs(dataset_path, description):
        """辅助函数：扫描指定路径下的数字样本目录"""
        sample_dirs = []
        if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
            # 遍历 .../train/ 或 .../val/ 目录下的所有子文件夹
            for item_name in sorted(os.listdir(dataset_path)): # sorted 确保顺序一致
                item_path = os.path.join(dataset_path, item_name)
                # 检查是否是目录，并且目录名是数字 (符合 1, 2, 3... 规范)
                if os.path.isdir(item_path) and item_name.isdigit():
                    # 将 "样本组" 的 *完整路径* 添加到列表中
                    sample_dirs.append(item_path)
            if not sample_dirs:
                 print(f"[Warning] 在 {description} 目录 '{dataset_path}' 中未找到数字命名的样本子目录。", file=sys.stderr)
        else:
            print(f"[Warning] {description} 目录 '{dataset_path}' 不存在或不是一个目录。", file=sys.stderr)
        return sample_dirs

    # 扫描训练集
    # args.train_data_dir_list = scan_sample_dirs(args.train_dataset_path, "训练集 (train)")
    
    # 扫描验证集
    args.test_data_dir_list = scan_sample_dirs(args.test_dataset_path, "验证集 (val)")
    
    # if hasattr(args, 'train_data_name_list'): del args.train_data_name_list
    # if hasattr(args, 'test_data_name_list'): del args.test_data_name_list
    # if hasattr(args, 'train_data_list_path'): del args.train_data_list_path
    # if hasattr(args, 'test_data_list_path'): del args.test_data_list_path
    
    # with open(args.train_data_list_path, "r") as f:
    #     # data_name_list = f.read() 
    #     data_name_list = [data_name.strip() for data_name in f]
    # args.train_data_name_list = data_name_list

    # with open(args.test_data_list_path, "r") as f:
    #     # data_name_list = f.read()
    #     test_data_name_list = [data_name.strip() for data_name in f]
    # args.test_data_name_list = test_data_name_list
    
    trainer = Trainer(args)
    trainer.validation()


if __name__ == "__main__":
    main()
