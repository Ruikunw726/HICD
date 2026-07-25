import  cv2
import numpy as np
import os
from typing import Tuple
from .io_utils import read_geotiff

def generate_aligned_chessboard(img1, img2, mkpts0, mkpts1, patch_size=32, save_path=None, max_size=10000):
    """
    根据匹配点生成仿射变换矩阵，将图像2进行变换和图像1对齐后，
    将两幅图像切成小patch并间隔拼接成棋盘格图像，可选择保存该图像。

    参数:
    img1 (numpy.ndarray): 第一张图像
    img2 (numpy.ndarray): 第二张图像
    mkpts0 (numpy.ndarray): 第一张图像上的匹配点
    mkpts1 (numpy.ndarray): 第二张图像上的匹配点
    patch_size (int): 小patch的尺寸，默认为32
    save_path (str): 保存棋盘格图像的文件路径，若为None则不保存
    max_size (int): 图像最大尺寸限制，默认为10000

    返回:
    numpy.ndarray: 拼接后的棋盘格图像
    """
    # 检查匹配点数量
    if len(mkpts0) < 3 or len(mkpts1) < 3:
        print(f"Warning: 匹配点数量太少 ({len(mkpts0)} 个)，无法进行可靠的仿射变换")
        return img1  # 返回原图像
    
    try:
        # 确保输入图像是2D的
        if len(img1.shape) == 3:
            img1 = img1 if img1.shape[2] <= img1.shape[0] else img1[:, :, 0]
        if len(img2.shape) == 3:
            img2 = img2 if img2.shape[2] <= img2.shape[0] else img2[:, :, 0]
        
        print(f"图像1尺寸: {img1.shape}")
        print(f"图像2尺寸: {img2.shape}")
        
        h, w = img1.shape[:2]
        
        # 检查图像尺寸是否超过OpenCV限制
        if h > 32000 or w > 32000:
            print(f"Warning: 图像尺寸过大 ({h}x{w})，将进行降采样")
            # 计算缩放因子
            scale = min(max_size / h, max_size / w, 1.0)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # 降采样图像
            img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 相应地缩放匹配点坐标
            mkpts0_scaled = mkpts0 * scale
            mkpts1_scaled = mkpts1 * scale
            
            print(f"降采样后尺寸: {img1.shape}")
            h, w = new_h, new_w
        else:
            mkpts0_scaled = mkpts0
            mkpts1_scaled = mkpts1
        
        # 计算仿射变换矩阵
        M, _ = cv2.estimateAffinePartial2D(mkpts1_scaled, mkpts0_scaled)
        
        if M is None:
            print("Warning: 无法计算仿射变换矩阵，返回原图像")
            return img1
        
        print(f"准备进行仿射变换，目标尺寸: {w}x{h}")
        
        # 使用仿射变换矩阵将图像2进行变换
        img2_aligned = cv2.warpAffine(img2, M, (w, h), flags=cv2.INTER_LINEAR)
        
        # 确保数据类型一致
        if img1.dtype != img2_aligned.dtype:
            img2_aligned = img2_aligned.astype(img1.dtype)
        
        # 创建棋盘格图像
        chessboard = np.zeros_like(img1)
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                end_i = min(i + patch_size, h)
                end_j = min(j + patch_size, w)
                
                if (i // patch_size + j // patch_size) % 2 == 0:
                    chessboard[i:end_i, j:end_j] = img1[i:end_i, j:end_j]
                else:
                    chessboard[i:end_i, j:end_j] = img2_aligned[i:end_i, j:end_j]

        # 保存棋盘格图像
        if save_path is not None:
            success = cv2.imwrite(save_path, chessboard)
            if success:
                print(f"棋盘格图像已保存到: {save_path}")
            else:
                print(f"保存失败，尝试使用不同的文件格式")
                # 尝试保存为提法格式
                tif_path = save_path.rsplit('.', 1)[0] + '.tif'
                success = cv2.imwrite(tif_path, chessboard)
                if success:
                    print(f"棋盘格图像已保存为tif格式: {tif_path}")

        return chessboard
    
    except Exception as e:
        print(f"生成棋盘格时出错: {e}")
        import traceback
        traceback.print_exc()
        return img1


# 修改调用部分的代码
# 修改调用部分的代码
def generate_chessboard_with_fix(img1, output_path, P1, P2, chessboard_path=None, generate_chessboard=True, patch_size=32):
    """修复后的棋盘格生成函数"""
    if not generate_chessboard:
        return
        
    try:
        if chessboard_path is None:
            base_dir = os.path.dirname(output_path)
            chessboard_path = os.path.join(base_dir, "chessboard.tif")  # 改为tif格式
        
        # 处理参考影像数据
        ref_img_data = img1['data']
        print(f"参考影像原始形状: {ref_img_data.shape}")
        
        # 正确处理多维数组
        if len(ref_img_data.shape) == 3:
            if ref_img_data.shape[0] < ref_img_data.shape[2]:  # (C, H, W) 格式
                ref_img_data = ref_img_data[0]  # 取第一个波段
            else:  # (H, W, C) 格式
                ref_img_data = ref_img_data[:, :, 0]  # 取第一个波段
        
        print(f"处理后参考影像形状: {ref_img_data.shape}")
        
        # 读取配准后的影像
        aligned_img = read_geotiff(output_path)
        aligned_img_data = aligned_img['data']
        print(f"配准影像原始形状: {aligned_img_data.shape}")
        
        # 正确处理配准后的影像
        if len(aligned_img_data.shape) == 3:
            if aligned_img_data.shape[0] < aligned_img_data.shape[2]:  # (C, H, W) 格式
                aligned_img_data = aligned_img_data[0]  # 取第一个波段
            else:  # (H, W, C) 格式
                aligned_img_data = aligned_img_data[:, :, 0]  # 取第一个波段
        
        print(f"处理后配准影像形状: {aligned_img_data.shape}")
        
        # 确保两个图像大小相同
        h, w = ref_img_data.shape[:2]
        if aligned_img_data.shape != ref_img_data.shape:
            aligned_img_data = cv2.resize(aligned_img_data, (w, h), interpolation=cv2.INTER_LINEAR)
            print(f"配准影像已调整尺寸到: {aligned_img_data.shape}")
        
        # 生成棋盘格
        chessboard = generate_aligned_chessboard(
            ref_img_data, aligned_img_data, 
            P1, P2, 
            patch_size=64,  # 可以根据需要调整
            save_path=chessboard_path,
            max_size=8000  # 限制最大尺寸
        )
        
        if chessboard is not None:
            print(f"棋盘格图像生成成功，尺寸: {chessboard.shape}")
        
    except Exception as e:
        print(f"生成棋盘格过程中出错: {e}")
        import traceback
        traceback.print_exc()