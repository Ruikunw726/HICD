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

from HICD.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from HICD.changedetection.datasets.make_data_loader_pure import make_data_loader,DamageDataset_train
from HICD.changedetection.utils_func.metrics import Evaluator
from HICD.changedetection.models.STMambaBDA import STMambaBDA

import HICD.changedetection.utils_func.lovasz_loss as L


class Trainer(object):
    def __init__(self, args):
        
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)

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
            # for k, v in checkpoint.items():
            #     if k in state_dict:
            #         model_dict[k] = v
                    
            exclude_keys = ["aux_clf.", "main_clf."]  # 需排除的层前缀
            filtered_checkpoint = {
                k: v for k, v in checkpoint.items()
                if not any(ex in k for ex in exclude_keys)  # 跳过分类层
                }
            # odel_dict = {}
            for k, v in filtered_checkpoint.items():
                if k in state_dict and v.shape == state_dict[k].shape:  # 检查键名和维度
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


    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        
        self.start_time = time.time()

        train_enumerator = enumerate(self.train_data_loader)
        
        total_epochs = self.args.epochs
        
        for epoch in range(self.args.epochs):
            
            self.deep_model.train()
            epoch_train_losses = []
            
            elem_num = len(self.train_data_loader)
            if elem_num == 0:
                print(f"[ERROR] Epoch {epoch}: train_data_loader is empty. Skipping.", file=sys.stderr)
                continue
            for itera, data in enumerate(self.train_data_loader):
                
                
                pre_change_imgs, post_change_imgs,labels_clf, _,_,_ = data
                
                
                pre_change_imgs = pre_change_imgs.cuda()
                
                post_change_imgs = post_change_imgs.cuda()
                
                labels_clf = labels_clf.cuda().long()
                
                # labels_clf = torch.where(labels_clf == 2, torch.tensor(1).cuda(), labels_clf)
                # labels_clf = torch.clamp(labels_clf, 0, 1)
                
                # class_weights = torch.tensor([1.0, 500.0, 1.0]).cuda()
    
                # labels_clf[labels_clf == 0] = 255
                
                output_clf = self.deep_model(pre_change_imgs, post_change_imgs)
    
                self.optim.zero_grad()
                
                # ce_loss_loc = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
                # lovasz_loss_loc = L.lovasz_softmax(F.softmax(output_loc, dim=1), labels_loc, ignore=255)
                
                # ce_loss_clf = F.cross_entropy(output_clf, labels_clf, weight=class_weights , ignore_index=255)
                ce_loss_clf = F.cross_entropy(output_clf, labels_clf, ignore_index=255)
                
                lovasz_loss_clf = L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_clf, ignore=255)
                final_loss =  ce_loss_clf + 0.75 * lovasz_loss_clf
                
                epoch_train_losses.append(final_loss.item())
                # final_loss = ce_loss_loc + lovasz_loss_loc
    
                final_loss.backward()
    
                self.optim.step()
                
                avg_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else 0.0
                
            if total_epochs > 0:
                # (epoch + 1) 是已经完成的epoch数，total_epochs是总epoch数
                percentage = int(((epoch + 1) / total_epochs) * 100)
                # 按照指定格式打印进度
                print(f"Progress:{percentage}%")
                # 立即刷新标准输出，确保外部进程能实时捕获
                sys.stdout.flush()
                
                # 该轮epoch验证阶段：
    
            self.deep_model.eval()
                
            avg_val_loss, oaf1,_,_, damage_f1_score = self.validation()
    
            self.save(epoch)
                
            log_data = {
                    "epoch": epoch + 1,
                    "time": float(time.time() - self.start_time),
                    "learning rate": self.args.learning_rate,
                    "loss": avg_train_loss,
                    "train loss": avg_train_loss,
                    "val loss": avg_val_loss,
                    # "loc_f1": (移除)
                    "class1_f1":oaf1[0], # (你的主要指标)
                    "class2_f1":oaf1[1],
                    "class3_f1":oaf1[2],
                    "class4_f1":oaf1[3]
                    
            }
            print(json.dumps(log_data))
            
            log_file_path = self.args.training_log
                
            try:
                # 使用 'a' 模式，这样每次调用都会在文件末尾添加新行
                # encoding='utf-8' 确保中文或其他特殊字符正确写入
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    # 将字典转换为 JSON 字符串并写入，加上换行符
                    log_file.write(json.dumps(log_data) + '\n')
            except Exception as e:
                # 如果写入失败，在 stderr 打印错误信息，避免程序崩溃
                print(f"[ERROR] Epoch {epoch}: 无法写入训练日志到 {log_file_path}. Error: {e}", file=sys.stderr)

    def validation(self):
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        dataset = DamageDataset_train(self.args.val_data_dir_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False)
        torch.cuda.empty_cache()
        
        epoch_val_losses = [] # [新增] 用于计算平均验证损失
        
        # vbar = tqdm(val_data_loader, ncols=50)
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels_clf, _,_,_ = data

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
                
        avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else 0.0
        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score,precision,recall = self.evaluator_clf.Damage_F1_socore()
        # harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        oaf1 = damage_f1_score
        
        
        # print(f'lofF1 is {loc_f1_score}, clfF1 is {harmonic_mean_f1}, oaF1 is {oaf1}, '
        #       f'sub class F1 score is {damage_f1_score}')
        # print(f'precision is {precision}, recall is {recall}')
        return avg_val_loss, oaf1, precision, recall, damage_f1_score


def main():
    parser = argparse.ArgumentParser(description="Training on test dataset")
    parser.add_argument('--cfg', type=str, default=r'./HICD/changedetection/configs/vssm1/vssm_small_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)

    parser.add_argument('--dataset', type=str, default='test')
    parser.add_argument('--type', type=str, default='train')
    
    parser.add_argument('--input_data', type=str, default=r'/home/user/桌面/train/')
    
   
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list) 
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=16000)
    parser.add_argument('--model_type', type=str, default='STMambaBDA')
    parser.add_argument('--model_param_path', type=str, default='/home/user/桌面/results/saved_models')
    
    parser.add_argument('--training_log', type=str, default='/home/user/桌面/results/saved_models/training_log.txt')
    
    parser.add_argument('--pure_inference', type=bool, default=False)

    parser.add_argument('--resume', type=str,
        # default=r'E:/baidu_download/Changemamba/HICD/changedetection/saved_models/xBD/MambaBDA_Small_1743787146.0769622/37500_model.pth')
        default=None
        # default=r'./HICD/changedetection/saved_models/test/199.pth'
        )
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)

    args = parser.parse_args()
    
    args.train_dataset_path = os.path.join(args.input_data, 'train') # 规范
    args.val_dataset_path = os.path.join(args.input_data, 'val')
    
    
    
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
    args.train_data_dir_list = scan_sample_dirs(args.train_dataset_path, "训练集 (train)")
    
    # 扫描验证集
    args.val_data_dir_list = scan_sample_dirs(args.val_dataset_path, "验证集 (test)")
    
    if hasattr(args, 'train_data_name_list'): del args.train_data_name_list
    if hasattr(args, 'test_data_name_list'): del args.test_data_name_list
    if hasattr(args, 'train_data_list_path'): del args.train_data_list_path
    if hasattr(args, 'test_data_list_path'): del args.test_data_list_path
    
    # with open(args.train_data_list_path, "r") as f:
    #     # data_name_list = f.read() 
    #     data_name_list = [data_name.strip() for data_name in f]
    # args.train_data_name_list = data_name_list

    # with open(args.test_data_list_path, "r") as f:
    #     # data_name_list = f.read()
    #     test_data_name_list = [data_name.strip() for data_name in f]
    # args.test_data_name_list = test_data_name_list
    
    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
