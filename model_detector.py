import sys
from pathlib import Path
FILE = Path(__file__).resolve()


ROOT1 = FILE.parents[1]
ROOT2 = FILE.parents[2]

if str(ROOT1) not in sys.path:
    sys.path.append(str(ROOT1))
if str(ROOT2) not in sys.path:
    sys.path.append(str(ROOT2))

import os
os.environ['PROJ_LIB']='/usr/share/proj'
os.environ['GDAL_DATA']='/usr/share/gdal'
os.environ['GTIFF_SRS_SOURCE']='EPSG'

import argparse
from changedetection.script import train_MambaBDA_test,infer_MambaBDA_pure,train_MambaBDA_eval
import torch
import time
import json


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


def setup_parsers():
    """设置所有模式的 argparse 子命令解析器。"""
    parser = argparse.ArgumentParser(description="模型训练、评估、推理工具")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="选择运行模式：train/eval/infer")

    # --- 1. Train 模式 ---
    parser_train = subparsers.add_parser('train', help='Run training mode')
    # 路径参数 (来自 algorithm_runner 的命令行)
    parser_train.add_argument('--input_data', type=str, required=True, help='Root path for input data')
    
    parser_train.add_argument('--output_data', type=str, required=True, help='Path to save models')
    parser_train.add_argument('--pretrained', type=str, default=None, help='Path to pretrained model weight')
    # parser_train.add_argument('--training_log', type=str, default=None, help='Path for training log output')

    # 超参数 (来自原 parse_train 和 config.json)
    parser_train.add_argument('--cfg', type=str, default='./MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml')
    parser_train.add_argument("--opts", help="Modify config options.", default=None, nargs='+')
    parser_train.add_argument('--pretrained_weight_path', type=str)
    parser_train.add_argument('--dataset', type=str, default='test')
    parser_train.add_argument('--type', type=str, default='train')
    parser_train.add_argument('--train_dataset_path', type=str)
    parser_train.add_argument('--train_data_list_path', type=str)
    parser_train.add_argument('--test_dataset_path', type=str)
    parser_train.add_argument('--test_data_list_path', type=str)
    parser_train.add_argument('--shuffle', type=bool, default=True)
    parser_train.add_argument('--batch_size', type=int, default=6)
    parser_train.add_argument('--crop_size', type=int, default=256)
    parser_train.add_argument('--start_iter', type=int, default=0)
    parser_train.add_argument('--cuda', type=bool, default=True)
    parser_train.add_argument('--max_iters', type=int, default=36000)
    parser_train.add_argument('--epochs', type=int, default=50)
    parser_train.add_argument('--model_type', type=str, default='STMambaBDA')
    parser_train.add_argument('--model_param_path', type=str, default='../saved_models')
    parser_train.add_argument('--resume', type=str, default=None) # runner 会传入
    parser_train.add_argument('--learning_rate', type=float, default=0.0005)
    parser_train.add_argument('--momentum', type=float, default=0.9)
    parser_train.add_argument('--weight_decay', type=float, default=0.005)
    
    parser_train.add_argument('--training_log', type=str, default='../saved_models/training_log.txt')
    
    # --- 2. Infer/Eval 模式 (假设 eval 参数与 infer 相同，都使用 infer 的参数集) ---
    parser_infer = subparsers.add_parser('infer', help='Run inference mode')
    parser_eval = subparsers.add_parser('eval', help='Run evaluation mode')
    
    for p in [parser_infer]:
        # 路径参数
        p.add_argument('--pretrained', type=str, default=None, help='Path to pretrained model weight')
        
        p.add_argument('--input_data', type=str, required=True, help='Root path for input pre data')
        p.add_argument('--input_data2', type=str, required=True, help='Root path for input post data')
        
        p.add_argument('--output_data', type=str, required=True, help='Path to save results')
        
        # 额外的 runner 参数 (来自 config.json)
        p.add_argument('--metaxml_path', type=str, default=None, help='--inputMetaArg1')
        p.add_argument('--slice', type=str, default=None, help='--sliceArg1')
        p.add_argument('--inputXMLArg1', type=str, default=None, help='--inputXMLArg1')
        
        # 内部参数
        p.add_argument('--cfg', type=str, default='./MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml')
        p.add_argument("--opts", default=None, nargs='+')
        p.add_argument('--pretrained_weight_path', type=str)
        p.add_argument('--dataset', type=str, default='test')
        p.add_argument('--type', type=str, default='test')
        p.add_argument('--test_dataset_path', type=str)
        p.add_argument('--test_data_list_path', type=str)
        p.add_argument('--shuffle', type=bool, default=True)
        p.add_argument('--batch_size', type=int, default=1)
        p.add_argument('--crop_size', type=int, default=256)
        p.add_argument('--start_iter', type=int, default=0)
        p.add_argument('--cuda', type=bool, default=True)
        p.add_argument('--max_iters', type=int, default=10000)
        p.add_argument('--model_type', type=str, default='MambaBDA')
        p.add_argument('--result_saved_path', type=str, default='../results')
        p.add_argument('--resume', type=str, default='./MambaCD/changedetection/saved_models/test/data_test_model.pth')
        p.add_argument('--learning_rate', type=float, default=1e-4)
        p.add_argument('--momentum', type=float, default=0.9)
        p.add_argument('--weight_decay', type=float, default=5e-4)
    
    
    parser_eval.add_argument('--input_data', type=str, required=True, help='Root path for input data')
    
    parser_eval.add_argument('--output_data', type=str, required=True, help='Path to save models')
    parser_eval.add_argument('--pretrained', type=str, default=None, help='Path to pretrained model weight')
    # parser_train.add_argument('--training_log', type=str, default=None, help='Path for training log output')

    # 超参数 (来自原 parse_train 和 config.json)
    parser_eval.add_argument('--cfg', type=str, default='./MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml')
    parser_eval.add_argument("--opts", help="Modify config options.", default=None, nargs='+')
    parser_eval.add_argument('--pretrained_weight_path', type=str)
    parser_eval.add_argument('--dataset', type=str, default='test')
    parser_eval.add_argument('--type', type=str, default='train')
    parser_eval.add_argument('--train_dataset_path', type=str)
    parser_eval.add_argument('--train_data_list_path', type=str)
    parser_eval.add_argument('--test_dataset_path', type=str)
    parser_eval.add_argument('--test_data_list_path', type=str)
    parser_eval.add_argument('--shuffle', type=bool, default=True)
    parser_eval.add_argument('--batch_size', type=int, default=1)
    parser_eval.add_argument('--crop_size', type=int, default=256)
    parser_eval.add_argument('--start_iter', type=int, default=0)
    parser_eval.add_argument('--cuda', type=bool, default=True)
    parser_eval.add_argument('--max_iters', type=int, default=36000)
    parser_eval.add_argument('--epochs', type=int, default=50)
    parser_eval.add_argument('--model_type', type=str, default='STMambaBDA')
    parser_eval.add_argument('--model_param_path', type=str, default='../saved_models')
    parser_eval.add_argument('--resume', type=str, default=None) # runner 会传入
    parser_eval.add_argument('--learning_rate', type=float, default=0.0005)
    parser_eval.add_argument('--momentum', type=float, default=0.9)
    parser_eval.add_argument('--weight_decay', type=float, default=0.005)
    

    return parser


def parse_train(args):
    """
    此函数现在接收 argparse 对象，并执行数据路径构造和列表读取。
    """
    # 1. 构造内部路径（使用 runner 传入的 --input_data 根目录）
    args.train_dataset_path = os.path.join(args.input_data, 'train') # 规范
    args.val_dataset_path = os.path.join(args.input_data, 'val')
    
    # 2. 设置输出路径
    args.model_param_path = args.output_data
    
    # 3. 处理 resume 路径 (原代码中 hard-coded 的默认值逻辑复杂，这里简化为 runner 传入优先)
    # 如果 runner 传入了 --pretrain，则覆盖 args.resume
    if args.pretrained:
        args.resume = args.pretrained
    # 如果 --resume 没被命令行指定，使用默认值
    # if not args.resume:
        # args.resume = './MambaCD/changedetection/saved_models/test/data_test_model.pth'
        
    
    

    # 扫描训练集
    args.train_data_dir_list = scan_sample_dirs(args.train_dataset_path, "训练集 (train)")
    
    # 扫描验证集
    args.val_data_dir_list = scan_sample_dirs(args.val_dataset_path, "验证集 (val)")
        
    if not args.train_data_dir_list:
        print(f"[ERROR] 在训练集目录 '{args.train_dataset_path}' 中未找到任何有效的训练样本组目录。任务无法继续。", file=sys.stderr)
        sys.exit(1) # 如果没有训练数据，必须失败退出

    #移除旧的 .txt 列表属性，避免混淆
    if hasattr(args, 'train_data_name_list'): del args.train_data_name_list
    if hasattr(args, 'test_data_name_list'): del args.test_data_name_list
    if hasattr(args, 'train_data_list_path'): del args.train_data_list_path
    if hasattr(args, 'test_data_list_path'): del args.test_data_list_path

        
    return args

def parse_infer(args):
    """
    此函数现在接收 argparse 对象，并执行推理数据路径构造和列表读取。
    """
    
    # 2. 设置结果保存路径
    args.result_saved_path = args.output_data

    # 3. 处理 resume 路径 (使用 --pretrain 覆盖默认 resume 路径)
    if args.pretrained:
        args.resume = args.pretrained
    
        
    return args

def parse_eval(args):
    """准备 eval 模式所需的 args 属性 (扫描 test 目录)"""
    args.test_dataset_path = os.path.join(args.input_data, 'test')
    args.result_saved_path = args.output_data # 映射给下游脚本
    args.resume = args.pretrained           # 映射给下游脚本

    args.test_data_dir_list = scan_sample_dirs(args.test_dataset_path, "评估集 (test)")

    if not args.test_data_dir_list:
        print(f"[ERROR] 在评估集目录 '{args.test_dataset_path}' 中未找到任何有效的样本组目录。", file=sys.stderr)
        # sys.exit(1) # 让下游脚本处理退出
        # 返回 False 或 None 表示准备失败
        return None
    args.model_param_path = args.output_data

    # 清理可能冲突的属性
    # if hasattr(args, 'train_data_dir_list'): del args.train_data_dir_list
    # if hasattr(args, 'val_data_dir_list'): del args.val_data_dir_list

    return args

def train(args):
    
    try:
    
        args=parse_train(args)
        
        
        trainer = train_MambaBDA_test.Trainer(args)
        trainer.training()
        
        print("Training success")
        print("Task Finished")
        torch.cuda.empty_cache()
        
        return 0
    
    except Exception as e:
        print(f"Training failed: {str(e)}")

        print("Task Finished")

        torch.cuda.empty_cache()
        
        sys.exit(-1)
    

    
def infer(args):
    
    inference_start = time.time()
    
    args=parse_infer(args)
    
    base_dir = os.path.dirname(args.result_saved_path)
    os.makedirs(base_dir, exist_ok=True)
    
    # 这里的名字可以起个带有 temp 标记的
    temp_registered_filename = "temp_registered.tif"
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
        
        has_temp_file = True
        
    except Exception as e:
        print(f"regis failed: {e}")
        print("using oringinal images to continue...")
    
    try:
        
        trainer = infer_MambaBDA_pure.Trainer(args)
        
        trainer.infer()
        
        print("infer success")
        print("Task Finished")
        torch.cuda.empty_cache()
        
        return 0
    
    except Exception as e:
        print(f"infer failed: {str(e)}")
        print("Task Finished")

        torch.cuda.empty_cache()
        sys.exit(-1)
    
    # finally:
    #     # =====================================================================
    #     # [Stage 2] 自动清理垃圾
    #     # =====================================================================
    #     # 只有当我们确实生成了临时文件，且该文件还在硬盘上时，才删除
    #     if has_temp_file and os.path.exists(temp_registered_path):
    #         try:
    #             os.remove(temp_registered_path)
    #         except Exception as e:
    #             print(f"warning: temp files cannot be deleted: {e}")
    #             print(f"please delete manually: {temp_registered_path}")
    #     if has_temp_cropfile:
    #         for p in aligned_paths:
    #             if os.path.exists(p):
    #                 try:
    #                     os.remove(p)
    #                 except Exception as e:
    #                     print(f"delete crop files failed:{e}")
    #                     print(f"please delete manually:{p}")
    inference_end = time.time()
    print(f"time cost:{inference_start-inference_end:.2f}s")
        
def eval1(args):
    
    try:
        args=parse_eval(args)
        
        trainer = train_MambaBDA_eval.Trainer(args)
        
        trainer.validation()
        
        print("eval success!")
        print("Task Finished")
        torch.cuda.empty_cache()
        
        return 0
    
    except Exception as e:
        print(f"eval failed: {str(e)}")
        print("Task Finished")

        torch.cuda.empty_cache()
        sys.exit(-1)
    

def main():

    parser = setup_parsers()

    args = parser.parse_args() 

    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        infer(args)
    elif args.mode == 'eval':
        eval1(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()



