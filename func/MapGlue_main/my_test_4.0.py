import pandas as pd
import os
from pathlib import Path
from utils.main import register_images


my_path = "/app/datatemp/test.csv"  # CSV文件路径
# 方式1：单个图像配准（原有功能）
single_mode = False  # 设置为True使用单个配准模式


def process_csv_batch(csv_file_path, 
                     grid_step_deg=0.1,
                     model_path='./weights/fastmapglue_model.pt',
                     num_keypoints=1024,
                     save_control_points=True,
                     generate_chessboard=False):
    """
    批量处理CSV文件中的图像配准任务
    
    参数:
    csv_file_path: CSV文件路径
    其他参数与register_images函数相同
    
    CSV格式要求:
    第1列: reference_image - 参考图像路径
    第2列: target_image - 目标图像路径  
    第3列: output_image - 输出图像路径
    第4列: chessboard_path - 棋盘图路径
    """
    
    # 检查CSV文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误：CSV文件不存在: {csv_file_path}")
        return
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path)
        print(f"成功读取CSV文件，共{len(df)}行数据")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 检查CSV格式
    if df.shape[1] < 4:
        print(f"错误：CSV文件至少需要4列，当前只有{df.shape[1]}列")
        return
    
    # 处理每一行数据
    success_count = 0
    error_count = 0
    
    for index, row in df.iterrows():
        try:
            # 获取路径参数
            reference_image = str(row[0]).strip()
            target_image = str(row[1]).strip()
            output_image = str(row[2]).strip()
            chessboard_path = str(row[3]).strip()
            
            print(f"\n正在处理第{index + 1}行数据:")
            print(f"  参考图像: {reference_image}")
            print(f"  目标图像: {target_image}")
            print(f"  输出图像: {output_image}")
            print(f"  棋盘图路径: {chessboard_path}")
            
            # 检查输入文件是否存在
            if not os.path.exists(reference_image):
                print(f"  警告：参考图像不存在: {reference_image}")
                error_count += 1
                continue
                
            if not os.path.exists(target_image):
                print(f"  警告：目标图像不存在: {target_image}")
                error_count += 1
                continue
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_image)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"  创建输出目录: {output_dir}")
            
            chessboard_dir = os.path.dirname(chessboard_path)
            if chessboard_dir and not os.path.exists(chessboard_dir):
                os.makedirs(chessboard_dir, exist_ok=True)
                print(f"  创建棋盘图目录: {chessboard_dir}")
            
            # 调用配准函数（使用位置参数）
            register_images(
                reference_image,
                target_image,
                output_image,
                grid_step_deg=grid_step_deg,
                model_path=model_path,
                num_keypoints=num_keypoints,
                save_control_points=save_control_points,
                generate_chessboard=generate_chessboard,
                chessboard_path=chessboard_path
            )
            
            success_count += 1
            print(f"  ✓ 第{index + 1}行处理成功")
            
        except Exception as e:
            error_count += 1
            print(f"  ✗ 第{index + 1}行处理失败: {e}")
            continue
    
    # 输出处理结果统计
    print(f"\n批量处理完成:")
    print(f"  成功处理: {success_count} 个任务")
    print(f"  处理失败: {error_count} 个任务")
    print(f"  总计: {len(df)} 个任务")

if __name__ == "__main__":
    # # 方式1：单个图像配准（原有功能）
    # single_mode = False  # 设置为True使用单个配准模式
    
    if single_mode:
        # 单个图像配准示例
        reference_image = "/test_ESRI_18_4326.tif"
        target_image = "/sar_test.tif"
        output_image = "/affine/test_3.0.tif"

        register_images(
            reference_image,
            target_image,
            output_image,
            grid_step_deg=0.1,
            model_path='./weights/fastmapglue_model.pt',
            num_keypoints=1024,
            save_control_points=True,
            generate_chessboard=False,
            chessboard_path="/affine/chessboard.tif"
        )
    else:
        # 方式2：批量处理CSV文件
        csv_file_path = my_path  # CSV文件路径
        
        process_csv_batch(
            csv_file_path=csv_file_path,
            grid_step_deg=0.1,
            model_path='./weights/fastmapglue_model.pt',
            num_keypoints=1024,
            save_control_points=True,
            generate_chessboard=False
        )