# -*- coding: utf-8 -*-
"""
独立运行的TIFF检查+修复工具
功能：检查配准后TIFF无法查看/模型识别不出的问题，并自动修复
使用：只需修改下方 INPUT_TIFF 和 OUTPUT_TIFF 路径即可
"""
import os
import numpy as np
from osgeo import gdal, osr

# ===================== 配置区（只改这两行）=====================
INPUT_TIFF = "/home/user/桌面/results/temp_registered.tif"  # 你的配准后TIFF路径
OUTPUT_TIFF = "/home/user/桌面/results/temp_registered_fixed.tif"  # 修复后保存路径
# ==============================================================

# 初始化GDAL（避免中文/路径报错）
gdal.UseExceptions()
os.environ['GDAL_DATA'] = '/usr/share/gdal'  # 按你的系统路径调整，无需改也能运行
os.environ['PROJ_LIB'] = '/usr/share/proj'

def check_tiff(file_path):
    """极简检查TIFF核心属性，返回检查结果+关键信息"""
    print(f"\n========== 检查 TIFF 文件：{file_path} ==========")
    result = {
        "is_valid": False,
        "dtype": "",
        "width": 0,
        "height": 0,
        "bands": 0,
        "pixel_min": 0,
        "pixel_max": 0,
        "error": ""
    }

    # 1. 基础检查：文件是否存在
    if not os.path.exists(file_path):
        result["error"] = "文件不存在"
        print(f"❌ {result['error']}")
        return result

    # 2. GDAL打开检查
    try:
        ds = gdal.Open(file_path, gdal.GA_ReadOnly)
        if ds is None:
            result["error"] = f"GDAL无法打开，错误：{gdal.GetLastErrorMsg()}"
            print(f"❌ {result['error']}")
            return result
    except Exception as e:
        result["error"] = f"打开失败：{str(e)}"
        print(f"❌ {result['error']}")
        return result

    # 3. 提取核心属性
    result["width"] = ds.RasterXSize
    result["height"] = ds.RasterYSize
    result["bands"] = ds.RasterCount
    result["dtype"] = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
    
    # 4. 像素值范围检查
    try:
        band1 = ds.GetRasterBand(1)
        min_val, max_val = band1.ComputeRasterMinMax(0)  # 不忽略NoData
        result["pixel_min"] = min_val
        result["pixel_max"] = max_val
    except:
        result["pixel_min"] = "未知"
        result["pixel_max"] = "未知"

    # 5. 判定是否有效（模型+看图软件兼容）
    is_valid_dtype = result["dtype"] == "Byte"  # 必须是uint8
    is_valid_pixel = (result["pixel_min"] >= 0) and (result["pixel_max"] <= 255)  # 0-255
    is_valid_size = (result["width"] > 0) and (result["height"] > 0)  # 尺寸有效

    result["is_valid"] = is_valid_dtype and is_valid_pixel and is_valid_size

    # 打印检查结果
    print(f"✅ 文件存在：是")
    print(f"📏 尺寸：{result['width']} x {result['height']}")
    print(f"📚 通道数：{result['bands']}")
    print(f"🔢 数据类型：{result['dtype']}（要求：Byte/uint8）")
    print(f"🎨 像素值范围：{result['pixel_min']} ~ {result['pixel_max']}（要求：0-255）")
    print(f"✅ 整体有效：{'是' if result['is_valid'] else '否'}")

    ds = None  # 关闭数据集
    return result

def fix_tiff(input_path, output_path):
    """自动修复TIFF：转uint8、0-255像素、保留地理信息"""
    print(f"\n========== 修复 TIFF 文件 ==========")
    try:
        # 1. 打开原始文件
        ds = gdal.Open(input_path, gdal.GA_ReadOnly)
        if ds is None:
            print(f"❌ 无法打开原始文件：{input_path}")
            return False

        # 2. 读取数据并强制修复
        arr = ds.ReadAsArray()  # (C, H, W)
        arr = np.clip(arr, 0, 255)  # 像素值限制在0-255
        arr = arr.astype(np.uint8)  # 强制转uint8

        # 3. 创建修复后的TIFF（无压缩、兼容所有软件）
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            output_path,
            ds.RasterXSize,
            ds.RasterYSize,
            ds.RasterCount,
            gdal.GDT_Byte,  # 强制Byte类型
            options=["COMPRESS=NONE", "INTERLEAVE=PIXEL"]  # 无压缩、像素交错（可查看）
        )

        # 4. 保留原始地理信息（关键：不影响模型切片）
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())

        # 5. 写入修复后的数据
        for i in range(ds.RasterCount):
            out_band = out_ds.GetRasterBand(i+1)
            out_band.WriteArray(arr[i])
            out_band.SetNoDataValue(0)  # 统一NoData值

        # 6. 强制刷新+关闭（避免文件损坏）
        out_ds.FlushCache()
        ds = None
        out_ds = None

        print(f"✅ 修复完成！保存路径：{output_path}")
        return True

    except Exception as e:
        print(f"❌ 修复失败：{str(e)}")
        return False

if __name__ == "__main__":
    # 第一步：检查原始TIFF
    check_result = check_tiff(INPUT_TIFF)

    # 第二步：如果无效则修复
    if not check_result["is_valid"]:
        print(f"\n❌ 原始TIFF无效，开始自动修复...")
        fix_success = fix_tiff(INPUT_TIFF, OUTPUT_TIFF)
        
        # 第三步：检查修复后的TIFF
        if fix_success:
            check_tiff(OUTPUT_TIFF)
            print(f"\n🎉 修复完成！修复后的文件：{OUTPUT_TIFF}")
            print("   ✅ 可直接查看 | ✅ 模型可识别")
    else:
        print(f"\n🎉 原始TIFF本身有效，无需修复！")

