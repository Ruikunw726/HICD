import os
import glob
import random
import shutil
from tqdm import tqdm

def find_corresponding_image(basename, image_dir):
    """
    根据基础名称(basename)在指定目录中查找对应的影像文件。
    这允许影像和标签有不同的扩展名 (e.g., .png, .jpg, .tif)。
    """
    # 查找所有匹配 basename + ".*" 的文件
    candidates = glob.glob(os.path.join(image_dir, basename + ".*"))
    
    # 过滤掉非影像文件
    valid_images = [f for f in candidates if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg', '.tif', '.bmp']]
    
    if len(valid_images) == 0:
        return None
    elif len(valid_images) > 1:
        print(f"警告: 找到多个匹配 '{basename}' 的文件: {valid_images}。将使用第一个。")
        return valid_images[0]
    else:
        return valid_images[0]

def process_split(name_list, split_name, output_root_dir, src_paths):
    """
    处理一个分割（train 或 test）:
    1. 为每个样本创建 1, 2, 3... 文件夹
    2. 在其中创建 base_image, compare_image, label 文件夹
    3. 复制文件
    """
    
    # 确保 train 或 test 根目录存在
    os.makedirs(output_root_dir, exist_ok=True)
    
    print(f"\n正在处理 {split_name} 集，共 {len(name_list)} 个文件...")
    
    # 使用 tqdm 显示进度条
    # enumerate(..., start=1) 让我们的计数从 1 开始
    for i, basename in enumerate(tqdm(name_list, desc=f"创建 {split_name} 集"), start=1):
        
        # 1. 创建此样本的根文件夹 (e.g., .../train/1)
        sample_dir = os.path.join(output_root_dir, str(i))
        
        # 2. 创建三个子文件夹
        dest_base_dir = os.path.join(sample_dir, "base_image")
        dest_compare_dir = os.path.join(sample_dir, "compare_image")
        dest_label_dir = os.path.join(sample_dir, "label")
        
        os.makedirs(dest_base_dir, exist_ok=True)
        os.makedirs(dest_compare_dir, exist_ok=True)
        os.makedirs(dest_label_dir, exist_ok=True)

        # 3. 查找源文件
        # 标签文件 (我们知道扩展名是 .tif)
        src_label = os.path.join(src_paths['B_tif'], basename + ".tif")
        
        # 影像文件 (使用辅助函数查找，自动匹配扩展名)
        src_pre = find_corresponding_image(basename, src_paths['img_A'])
        src_post = find_corresponding_image(basename, src_paths['img_B'])
        
        # 4. 健全性检查
        if src_pre is None:
            print(f"警告: 在 '{src_paths['img_A']}' 中未找到 {basename} 对应的 pre 影像，跳过此样本。")
            continue
        if src_post is None:
            print(f"警告: 在 '{src_paths['img_B']}' 中未找到 {basename} 对应的 post 影像，跳过此样本。")
            continue
        if not os.path.exists(src_label):
            print(f"警告: 在 '{src_paths['B_tif']}' 中未找到 {basename}.tif 标签，跳过此样本。")
            continue
            
        # 5. 构建目标路径 (保持原始扩展名)
        dest_pre = os.path.join(dest_base_dir, os.path.basename(src_pre))
        dest_post = os.path.join(dest_compare_dir, os.path.basename(src_post))
        dest_label = os.path.join(dest_label_dir, os.path.basename(src_label))
        
        # 6. 复制文件
        try:
            shutil.copy2(src_pre, dest_pre)
            shutil.copy2(src_post, dest_post)
            shutil.copy2(src_label, dest_label)
        except Exception as e:
            print(f"错误: 复制文件 {basename} (到 {sample_dir}) 时出错: {e}")

def main():
    # --- 1. 配置路径 ---
    
    # 你的原始数据集根目录
    base_dir = "D:/code/测试影像/ZY"
    
    # 你的新数据集输出根目录
    output_dir = "D:/code/测试影像/dataset_split_ZY_v2" # 改个名字以免和旧的混淆
    
    # 分割比例
    train_ratio = 0.7
    
    # --- 2. 定义源文件夹路径 ---
    src_paths = {
        'img_A': os.path.join(base_dir, "img_A"),
        'img_B': os.path.join(base_dir, "img_B"),
        'B_tif': os.path.join(base_dir, "B_tif")
    }
    
    print(f"源数据文件夹:")
    print(f"  Base 影像 (img_A): {src_paths['img_A']}")
    print(f"  Compare 影像 (img_B): {src_paths['img_B']}")
    print(f"  标签 (B_tif):      {src_paths['B_tif']}")
    print(f"\n输出文件夹: {output_dir}")

    # --- 3. 定义目标根文件夹 ---
    output_train_root = os.path.join(output_dir, "train")
    output_test_root = os.path.join(output_dir, "test")

    # --- 4. 扫描文件并分割 ---
    print(f"\n正在从 '{src_paths['B_tif']}' 扫描标签文件...")
    
    try:
        label_files = [f for f in os.listdir(src_paths['B_tif']) if f.endswith('.tif')]
        all_basenames = [os.path.splitext(f)[0] for f in label_files]
    except FileNotFoundError:
        print(f"错误: 找不到标签文件夹: {src_paths['B_tif']}。请检查路径。")
        return
        
    if not all_basenames:
        print(f"错误: 在 '{src_paths['B_tif']}' 中未找到任何 .tif 标签文件。")
        return

    print(f"总共找到 {len(all_basenames)} 个标签文件。")

    # 随机打乱
    random.seed(42) # 使用固定种子确保每次运行结果可复现
    random.shuffle(all_basenames)

    # 计算分割点
    split_index = int(len(all_basenames) * train_ratio)
    
    # 分割列表
    train_names = all_basenames[:split_index]
    test_names = all_basenames[split_index:]

    print(f"分割比例: {train_ratio*100}% 训练 / {(1-train_ratio)*100:.0f}% 测试")
    print(f"训练集文件数: {len(train_names)}")
    print(f"测试集文件数: {len(test_names)}")

    # --- 5. 处理两个分割 ---
    # 处理训练集
    process_split(train_names, "train", output_train_root, src_paths)
    
    # 处理测试集
    process_split(test_names, "test", output_test_root, src_paths)

    print("\n--- 所有任务完成 ---")
    print(f"数据集已成功创建在: {output_dir}")

# --- 运行主函数 ---
if __name__ == "__main__":
    main()