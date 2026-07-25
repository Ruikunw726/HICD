import os
import shutil
import sys # 用于 exit
from pathlib import Path

# --- !!! 请在这里手动修改路径 !!! ---

# 源根目录：包含旧 'train' 文件夹的目录
# 例如: r"D:\旧数据集\我的项目"
SOURCE_ROOT_DIRECTORY = r"D:\MambaCD\frame_building"

# 目标根目录：你想要创建新 'train' 文件夹的地方
# 例如: r"D:\新数据集\符合规范"
DESTINATION_ROOT_DIRECTORY = r"D:\docker环境_2\test\MambaCD\frame_building"

# --- 路径修改结束 ---


def find_corresponding_file(base_name, directory, potential_suffixes):
    """
    在指定目录中查找与 base_name 匹配的文件。
    尝试 potential_suffixes 中定义的各种后缀。
    """
    for suffix in potential_suffixes:
        potential_filename = f"{base_name}{suffix}"
        potential_path = os.path.join(directory, potential_filename)
        if os.path.exists(potential_path):
            return potential_path, potential_filename # 返回完整路径和文件名
    return None, None

def restructure_dataset(source_root_dir, dest_root_dir):
    """
    将旧结构的数据集重组为符合 IIS 规范的新结构。

    Args:
        source_root_dir (str): 包含旧 'train' 文件夹的根目录。
        dest_root_dir (str):   用于存放新结构 'train' 文件夹的目标根目录。
    """
    print(f"源目录: {source_root_dir}")
    print(f"目标目录: {dest_root_dir}")

    # --- 1. 验证和定义源路径 ---
    if not os.path.isdir(source_root_dir):
        print(f"错误: 源根目录 '{source_root_dir}' 不存在或不是一个目录。请检查 SOURCE_ROOT_DIRECTORY 变量。")
        sys.exit(1)

    source_train_dir = os.path.join(source_root_dir, 'val')
    source_image_dir = os.path.join(source_train_dir, 'image')
    source_pre_dir = os.path.join(source_image_dir, 'pre')
    source_post_dir = os.path.join(source_image_dir, 'post')
    # [!! 修正 !!] clf 标签路径指向 train/label/clf
    source_label_dir = os.path.join(source_train_dir, 'label')
    source_clf_dir = os.path.join(source_label_dir, 'clf') # <--- 修正点

    # 检查源子路径是否存在
    if not os.path.isdir(source_pre_dir):
        print(f"错误: 源 'pre' 目录不存在: {source_pre_dir}")
        sys.exit(1)
    if not os.path.isdir(source_post_dir):
        print(f"错误: 源 'post' 目录不存在: {source_post_dir}")
        sys.exit(1)
    if not os.path.isdir(source_clf_dir): # <--- 修正点: 检查新的 clf 路径
        print(f"错误: 源 'clf' 目录 ('{source_clf_dir}') 不存在。请确认源结构是否为 train/label/clf/")
        sys.exit(1)

    # --- 2. 定义和创建目标路径 ---
    if not dest_root_dir or dest_root_dir == "请替换为你的目标根目录路径":
        print("错误: 目标根目录路径无效。请修改 DESTINATION_ROOT_DIRECTORY 变量。")
        sys.exit(1)

    dest_train_dir = os.path.join(dest_root_dir, 'val')
    os.makedirs(dest_train_dir, exist_ok=True) # 创建目标 train 目录
    print(f"将在以下位置创建新结构: {dest_train_dir}")

    # --- 3. 查找并处理 pre 图像 ---
    try:
        pre_files = sorted([
            f for f in os.listdir(source_pre_dir)
            if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('.')
        ])
    except FileNotFoundError:
        print(f"错误: 无法访问源 'pre' 目录: {source_pre_dir}")
        sys.exit(1)

    print(f"在 '{source_pre_dir}' 中找到 {len(pre_files)} 个 pre 图像文件。")

    sample_count = 0
    processed_count = 0

    # 定义可能的后缀（请根据你的实际文件名调整）
    pre_suffixes = ['_pre.tif', '_pre.tiff', '.tif', '.tiff']
    post_suffixes = ['_post.tif', '_post.tiff', '.tif', '.tiff']
    clf_suffixes = ['_clf.tif', '_clf.tiff', '_label.tif', '_label.tiff', '.tif', '.tiff']

    for pre_filename in pre_files:
        sample_count += 1
        print(f"\n处理样本 {sample_count}: {pre_filename}")

        # --- 4. 提取基础名 (保持之前的逻辑) ---
        base_name_found = False
        base_name = pre_filename
        pre_filename_lower = pre_filename.lower()
        for suffix in ['_pre.tif', '_pre.tiff', '_PRE.tif', '_PRE.tiff']:
            if pre_filename_lower.endswith(suffix):
                base_name = pre_filename[:-len(suffix)]
                base_name_found = True
                break
        if not base_name_found:
             if pre_filename_lower.endswith('.tif'): base_name = pre_filename[:-4]
             elif pre_filename_lower.endswith('.tiff'): base_name = pre_filename[:-5]

        if not base_name or base_name == pre_filename:
            print(f"  -> 警告: 无法从 '{pre_filename}' 提取基础名。将尝试使用全名 '{base_name}' 匹配。")

        print(f"  尝试使用的基础名: {base_name}")

        # --- 5. 查找对应的 post 和 clf 文件 ---
        pre_path = os.path.join(source_pre_dir, pre_filename)

        post_path, post_filename = find_corresponding_file(base_name, source_post_dir, post_suffixes)
        if post_path is None:
            print(f"  -> 警告: 未找到对应的 post 文件 (基础名: {base_name})。跳过此样本。")
            continue
        print(f"  找到 post 文件: {post_filename}")

        # [!! 修正 !!] 在正确的 source_clf_dir 中查找 clf 文件
        clf_path, clf_filename = find_corresponding_file(base_name, source_clf_dir, clf_suffixes) # <--- 修正点
        if clf_path is None:
            print(f"  -> 警告: 未找到对应的 clf/label 文件 (基础名: {base_name})。跳过此样本。")
            continue
        print(f"  找到 clf/label 文件: {clf_filename}")

        # --- 6. 创建目标目录结构 (保持不变) ---
        target_sample_dir = os.path.join(dest_train_dir, str(processed_count + 1))
        target_base_dir = os.path.join(target_sample_dir, 'base_image')
        target_compare_dir = os.path.join(target_sample_dir, 'compare_image')
        target_label_dir = os.path.join(target_sample_dir, 'label')

        os.makedirs(target_base_dir, exist_ok=True)
        os.makedirs(target_compare_dir, exist_ok=True)
        os.makedirs(target_label_dir, exist_ok=True)

        # --- 7. 复制文件 (目标路径不变，源 clf_path 已修正) ---
        try:
            dest_pre_path = os.path.join(target_base_dir, pre_filename)
            dest_post_path = os.path.join(target_compare_dir, post_filename)
            dest_clf_path = os.path.join(target_label_dir, clf_filename) # 使用找到的 clf 文件名

            print(f"  复制 '{pre_filename}' 到 '{dest_pre_path}'")
            shutil.copy2(pre_path, dest_pre_path)

            print(f"  复制 '{post_filename}' 到 '{dest_post_path}'")
            shutil.copy2(post_path, dest_post_path)

            print(f"  复制 '{clf_filename}' 到 '{dest_clf_path}'") # <--- 修正点: 使用正确的 clf_path
            shutil.copy2(clf_path, dest_clf_path)

            processed_count += 1

        except Exception as e:
            print(f"  -> 错误: 复制文件时出错: {e}。跳过此样本。")
            if os.path.exists(target_sample_dir):
                try: shutil.rmtree(target_sample_dir); print(f"    已删除部分创建的目录: {target_sample_dir}")
                except OSError as rm_error: print(f"    警告: 删除部分创建的目录失败: {rm_error}")

    print(f"\n处理完成。共扫描 {sample_count} 个 pre 文件，成功转换 {processed_count} 组样本。")
    print(f"新数据集结构已创建于: {dest_train_dir}")

if __name__ == "__main__":
    if SOURCE_ROOT_DIRECTORY == "请替换为你的源根目录路径" or not SOURCE_ROOT_DIRECTORY:
        print("错误: 请先修改脚本顶部的 SOURCE_ROOT_DIRECTORY 变量。")
    elif DESTINATION_ROOT_DIRECTORY == "请替换为你的目标根目录路径" or not DESTINATION_ROOT_DIRECTORY:
        print("错误: 请先修改脚本顶部的 DESTINATION_ROOT_DIRECTORY 变量。")
    else:
        restructure_dataset(SOURCE_ROOT_DIRECTORY, DESTINATION_ROOT_DIRECTORY)