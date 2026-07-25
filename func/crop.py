import numpy as np
import os
from osgeo import gdal,osr


def crop(img1,img2,out_dir):
    ds1 = gdal.Open(img1)
    ds2 = gdal.Open(img2)

    n_bands = ds1.RasterCount
    gt1 = ds1.GetGeoTransform()
    gt2 = ds2.GetGeoTransform()
    resx1=gt1[1]
    resy1=gt1[5]
    resx2=gt2[1]
    resy2=gt2[5]
    left1,top1=gt1[0],gt1[3]
    right1=left1 + ds1.RasterXSize * resx1
    bottom1 = top1 + ds1.RasterYSize * resy1

    left2,top2=gt2[0],gt2[3]
    right2=left2 + ds2.RasterXSize * resx2
    bottom2 = top2 + ds2.RasterYSize * resy2

    left = max(left1,left2)
    right = min(right1,right2)
    top = min(top1,top2)
    bottom = max(bottom1,bottom2)

    w = int((right-left)/resx1+0.5)
    h = int((top-bottom)/abs(resy1)+0.5)

    pre_crop = f"{out_dir}/temp_pre_crop.tif"

    gdal.Translate(pre_crop,ds1,
        projWin=[left,top,right,bottom],
    )
    ds1_crop=gdal.Open(pre_crop)

    post_crop = f"{out_dir}/temp_post_crop.tif"
    
    gdal.Translate(post_crop,ds2,
        projWin=[left,top,right,bottom],
    )   
    ds2_crop=gdal.Open(post_crop)

    arr1_mask = ds1_crop.GetRasterBand(1).ReadAsArray()
    arr2_mask = ds2_crop.GetRasterBand(1).ReadAsArray()
    if arr1_mask.shape != arr2_mask.shape:
        raise ValueError(f"crop size of two images is not correct,arr1:{arr1_mask.shape},arr2:{arr2_mask.shape}")

    mask = (arr1_mask!=0) & (arr2_mask!=0)

    driver = gdal.GetDriverByName('GTiff')
    h,w = arr1_mask.shape

    out1 = f"{out_dir}/temp_pre.tif"
    out2 = f"{out_dir}/temp_post.tif"

    dtype = gdal.GDT_UInt16
    d1 = driver.Create(out1,w,h,n_bands,dtype)
    d1.SetGeoTransform(ds1_crop.GetGeoTransform())
    d1.SetProjection(ds1_crop.GetProjection())
    for i in range(1,n_bands+1):
        arr = ds1_crop.GetRasterBand(i).ReadAsArray()
        arr_out = arr.copy()
        arr_out[~mask]=0
        d1.GetRasterBand(i).WriteArray(arr_out)
    d1.GetRasterBand(i).SetNoDataValue(0)
    d1 = None

    d2 = driver.Create(out2,w,h,n_bands,dtype)
    d2.SetGeoTransform(ds2_crop.GetGeoTransform())
    d2.SetProjection(ds2_crop.GetProjection())
    for i in range(1,n_bands+1):
        arr = ds2_crop.GetRasterBand(i).ReadAsArray()
        arr_out = arr.copy()
        arr_out[~mask]=0
        d2.GetRasterBand(i).WriteArray(arr_out)
    d2.GetRasterBand(i).SetNoDataValue(0)
    d2 = None
    
    ds1_crop = None
    ds2_crop = None
    if os.path.exists(pre_crop):
        os.remove(pre_crop)
    if os.path.exists(post_crop):
        os.remove(post_crop)
    return out1,out2




# def crop2(img1,img2,out_dir,crop_points=None):
#     ds1 = gdal.Open(img1)
#     ds2 = gdal.Open(img2)

#     n_bands = ds1.RasterCount
#     gt1 = ds1.GetGeoTransform()
#     gt2 = ds2.GetGeoTransform()
#     resx1=gt1[1]
#     resy1=gt1[5]
#     resx2=gt2[1]
#     resy2=gt2[5]
    
#     # img1_left = gt1[0]
#     # img1_right = gt1[0] + ds1.RasterXSize * resx1
#     # img1_top = gt1[3]
#     # img1_bottom = gt1[3] + ds1.RasterYSize * resy1
#     # print(f"===== 图像1真实范围（投影坐标/米） =====")
#     # print(f"左：{img1_left}, 右：{img1_right}, 上：{img1_top}, 下：{img1_bottom}")
#     # 新增：计算图像的经纬度范围
#     src_srs = osr.SpatialReference()
#     src_srs.ImportFromWkt(ds1.GetProjection())
#     dst_srs = osr.SpatialReference()
#     dst_srs.ImportFromEPSG(4326)  # WGS84经纬度
#     # 图像左上投影坐标转经纬度
#     # lon1, lat1, _ = osr.CoordinateTransformation(src_srs, dst_srs).TransformPoint(gt1[0], gt1[3])
#     # # 图像右下投影坐标转经纬度
#     # lon2, lat2, _ = osr.CoordinateTransformation(src_srs, dst_srs).TransformPoint(gt1[0]+ds1.RasterXSize*gt1[1], gt1[3]+ds1.RasterYSize*gt1[5])
#     # print(f"===== 图像实际覆盖的经纬度范围 ======")
#     # print(f"左上经纬度：{lon1}, {lat1}")
#     # print(f"右下经纬度：{lon2}, {lat2}")
    
#     if crop_points is not None:
        
#         transform = osr.CoordinateTransformation(dst_srs, src_srs)
        
#         # 3. 注意：crop_points传入的是 [经度, 纬度]，但TransformPoint需要 (纬度, 经度)！
#         transformed_points = []
#         for lon, lat in crop_points:
#             # 关键修正：TransformPoint(纬度, 经度) 而非 (经度, 纬度)
#             x, y, _ = transform.TransformPoint(lat, lon)
#             transformed_points.append([x, y])
        
#         x1,y1 = transformed_points[0]
#         x2,y2 = transformed_points[1]
#         x3,y3 = transformed_points[2]
#         x4,y4 = transformed_points[3]
        
#         left = min(x1,x2,x3,x4)
#         right = max(x1,x2,x3,x4)
#         top = max(y1,y2,y3,y4)
#         bottom = min(y1,y2,y3,y4)
#         # print(f"===== 转换后裁剪范围（投影坐标/米） =====")
#         # print(f"左：{left}, 右：{right}, 上：{top}, 下：{bottom}")
        
#         # inter_left = max(left, img1_left)
#         # inter_right = min(right, img1_right)
#         # inter_top = min(top, img1_top)  # 注意：投影坐标中top是数值大的一侧
#         # inter_bottom = max(bottom, img1_bottom)
#         # print(f"===== 裁剪范围与图像的交集 =====")
#         # print(f"交集左：{inter_left}, 交集右：{inter_right}, 交集上：{inter_top}, 交集下：{inter_bottom}")
        
#     else:
#         left1,top1=gt1[0],gt1[3]
#         right1=left1 + ds1.RasterXSize * resx1
#         bottom1 = top1 + ds1.RasterYSize * resy1
    
#         left2,top2=gt2[0],gt2[3]
#         right2=left2 + ds2.RasterXSize * resx2
#         bottom2 = top2 + ds2.RasterYSize * resy2
    
#         left = max(left1,left2)
#         right = min(right1,right2)
#         top = min(top1,top2)
#         bottom = max(bottom1,bottom2)

#     w = int((right-left)/resx1+0.5)
#     h = int((top-bottom)/abs(resy1)+0.5)

#     pre_crop = f"{out_dir}/temp_pre_crop.tif"

#     gdal.Translate(pre_crop,ds1,
#         projWin=[left,top,right,bottom],
#     )
#     ds1_crop=gdal.Open(pre_crop)

#     post_crop = f"{out_dir}/temp_post_crop.tif"
    
#     gdal.Translate(post_crop,ds2,
#         projWin=[left,top,right,bottom],
#     )   
#     ds2_crop=gdal.Open(post_crop)

#     arr1_mask = ds1_crop.GetRasterBand(1).ReadAsArray()
#     arr2_mask = ds2_crop.GetRasterBand(1).ReadAsArray()
#     if arr1_mask.shape != arr2_mask.shape:
#         raise ValueError(f"crop size of two images is not correct,arr1:{arr1_mask.shape},arr2:{arr2_mask.shape}")

#     mask = (arr1_mask!=0) & (arr2_mask!=0)

#     driver = gdal.GetDriverByName('GTiff')
#     h,w = arr1_mask.shape

#     out1 = f"{out_dir}/temp_pre.tif"
#     out2 = f"{out_dir}/temp_post.tif"

#     dtype = gdal.GDT_UInt16
#     d1 = driver.Create(out1,w,h,n_bands,dtype)
#     d1.SetGeoTransform(ds1_crop.GetGeoTransform())
#     d1.SetProjection(ds1_crop.GetProjection())
#     for i in range(1,n_bands+1):
#         arr = ds1_crop.GetRasterBand(i).ReadAsArray()
#         arr_out = arr.copy()
#         arr_out[~mask]=0
#         d1.GetRasterBand(i).WriteArray(arr_out)
#     d1.GetRasterBand(i).SetNoDataValue(0)
#     d1 = None

#     d2 = driver.Create(out2,w,h,n_bands,dtype)
#     d2.SetGeoTransform(ds2_crop.GetGeoTransform())
#     d2.SetProjection(ds2_crop.GetProjection())
#     for i in range(1,n_bands+1):
#         arr = ds2_crop.GetRasterBand(i).ReadAsArray()
#         arr_out = arr.copy()
#         arr_out[~mask]=0
#         d2.GetRasterBand(i).WriteArray(arr_out)
#     d2.GetRasterBand(i).SetNoDataValue(0)
#     d2 = None
    
#     ds1_crop = None
#     ds2_crop = None
#     if os.path.exists(pre_crop):
#         os.remove(pre_crop)
#     if os.path.exists(post_crop):
#         os.remove(post_crop)
#     return out1,out2

# def crop2(img1, img2, crop_points=None):
#     from osgeo import gdal, osr
#     import numpy as np

#     # 1. 读取数据（支持 文件路径 / 配准字典）
#     def get_full_data(img):
#         if isinstance(img, str):
#             ds = gdal.Open(img)
#             arr = ds.ReadAsArray()
#             geo = ds.GetGeoTransform()
#             crs = ds.GetProjection()
#             width = ds.RasterXSize
#             height = ds.RasterYSize
#             ds = None
#             return arr, geo, crs, width, height
#         # 读取配准后字典
#         arr = img['image']
#         geo = img['transform']
#         crs = img['crs']
#         width = img['width']
#         height = img['height']
#         return arr, geo, crs, width, height

#     # 获取两张图完整信息
#     arr1, gt1, crs1, w1, h1 = get_full_data(img1)
#     arr2, gt2, crs2, w2, h2 = get_full_data(img2)
#     n_bands = arr1.shape[0]
    
#     # ===================== 补上你原版的 resx1/resy1/resx2/resy2 =====================
#     resx1 = gt1[1]
#     resy1 = gt1[5]
#     resx2 = gt2[1]
#     resy2 = gt2[5]
#     # ==================================================================================

#     # ===================== 完全照搬你原版的坐标计算逻辑，一字不改！=====================
#     src_srs = osr.SpatialReference()
#     src_srs.ImportFromWkt(crs1)
#     dst_srs = osr.SpatialReference()
#     dst_srs.ImportFromEPSG(4326)

#     if crop_points is not None:
#         transform = osr.CoordinateTransformation(dst_srs, src_srs)
#         transformed_points = []
#         for lon, lat in crop_points:
#             # 完全保留你原版的经纬度转换
#             x, y, _ = transform.TransformPoint(lat, lon)
#             transformed_points.append([x, y])
        
#         x1,y1 = transformed_points[0]
#         x2,y2 = transformed_points[1]
#         x3,y3 = transformed_points[2]
#         x4,y4 = transformed_points[3]
        
#         left = min(x1,x2,x3,x4)
#         right = max(x1,x2,x3,x4)
#         top = max(y1,y2,y3,y4)
#         bottom = min(y1,y2,y3,y4)
#     else:
#         # 完全保留你原版的交集计算
#         left1,top1=gt1[0],gt1[3]
#         right1=left1 + w1 * resx1
#         bottom1 = top1 + h1 * resy1
    
#         left2,top2=gt2[0],gt2[3]
#         right2=left2 + w2 * resx2
#         bottom2 = top2 + h2 * resy2
    
#         left = max(left1,left2)
#         right = min(right1,right2)
#         top = min(top1,top2)
#         bottom = max(bottom1,bottom2)
#     # ==================================================================================

#     # 地理坐标 → 像素坐标（纯公式计算，不碰GDAL报错接口）
#     xoff = int((left - gt1[0]) / resx1 + 0.5)
#     yoff = int((top - gt1[3]) / resy1 + 0.5)
#     win_w = int((right - left) / resx1 + 0.5)
#     win_h = int((top - bottom) / abs(resy1) + 0.5)

#     # 边界保护（防止像素越界报错）
#     xoff = max(0, xoff)
#     yoff = max(0, yoff)
#     win_w = min(w1 - xoff, win_w)
#     win_h = min(h1 - yoff, win_h)

#     # 纯numpy像素裁剪（永不报错）
#     arr1_crop = arr1[:, yoff:yoff+win_h, xoff:xoff+win_w]
#     arr2_crop = arr2[:, yoff:yoff+win_h, xoff:xoff+win_w]

#     # 完全保留你原版的掩码处理
#     mask = (arr1_crop[0] != 0) & (arr2_crop[0] != 0)
#     for b in range(n_bands):
#         arr1_crop[b][~mask] = 0
#         arr2_crop[b][~mask] = 0

#     # 修正裁剪后的地理变换
#     new_gt1 = list(gt1)
#     new_gt1[0] += xoff * new_gt1[1]
#     new_gt1[3] += yoff * new_gt1[5]

#     new_gt2 = list(gt2)
#     new_gt2[0] += xoff * new_gt2[1]
#     new_gt2[3] += yoff * new_gt2[5]


#     # print(f"arr1_crop shape: {arr1_crop.shape}")
#     # print(f"arr2_crop shape: {arr2_crop.shape}")
#     # 返回标准字典
#     out1 = {
#         "image": arr1_crop,
#         "data": arr1_crop,
#         "transform": tuple(new_gt1),
#         "crs": crs1,
#         "height": win_h,
#         "width": win_w
#     }
#     out2 = {
#         "image": arr2_crop,
#         "data":arr2_crop,
#         "transform": tuple(new_gt2),
#         "crs": crs2,
#         "height": win_h,
#         "width": win_w
#     }

#     return out1, out2

# def crop2(img1, img2, crop_points=None):
#     from osgeo import gdal, osr
#     import numpy as np
#     import tempfile
#     import os

#     # 打开图像（支持路径 或 字典）
#     def open_img(img):
#         if isinstance(img, str):
#             return gdal.Open(img)
#         else:
#             # 从字典构造内存数据集
#             driver = gdal.GetDriverByName('MEM')
#             arr = img['image']
#             c, h, w = arr.shape
#             ds = driver.Create('', w, h, c, gdal.GDT_UInt16)
#             ds.SetGeoTransform(img['transform'])
#             ds.SetProjection(img['crs'])
#             for i in range(c):
#                 ds.GetRasterBand(i+1).WriteArray(arr[i])
#             return ds

#     ds1 = open_img(img1)
#     ds2 = open_img(img2)

#     gt1 = ds1.GetGeoTransform()
#     gt2 = ds2.GetGeoTransform()
#     resx1 = gt1[1]
#     resy1 = gt1[5]
#     resx2 = gt2[1]
#     resy2 = gt2[5]

#     # 坐标系
#     src_srs = osr.SpatialReference()
#     src_srs.ImportFromWkt(ds1.GetProjection())
#     dst_srs = osr.SpatialReference()
#     dst_srs.ImportFromEPSG(4326)

#     # 计算裁剪范围
#     if crop_points is not None:
#         transform = osr.CoordinateTransformation(dst_srs, src_srs)
#         transformed_points = []
#         for lon, lat in crop_points:
#             x, y, _ = transform.TransformPoint(lat, lon)
#             transformed_points.append([x, y])
#         x1, y1 = transformed_points[0]
#         x2, y2 = transformed_points[1]
#         x3, y3 = transformed_points[2]
#         x4, y4 = transformed_points[3]
#         left = min(x1, x2, x3, x4)
#         right = max(x1, x2, x3, x4)
#         top = max(y1, y2, y3, y4)
#         bottom = min(y1, y2, y3, y4)
#     else:
#         left1, top1 = gt1[0], gt1[3]
#         right1 = left1 + ds1.RasterXSize * resx1
#         bottom1 = top1 + ds1.RasterYSize * resy1

#         left2, top2 = gt2[0], gt2[3]
#         right2 = left2 + ds2.RasterXSize * resx2
#         bottom2 = top2 + ds2.RasterYSize * resy2

#         left = max(left1, left2)
#         right = min(right1, right2)
#         top = min(top1, top2)
#         bottom = max(bottom1, bottom2)

#     # ---------- 核心：用 GDAL 稳定裁剪（绝对不会裁错） ----------
#     with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
#         tmp_name = tmp.name

#     ds1_crop = gdal.Translate(tmp_name, ds1, projWin=[left, top, right, bottom])
#     arr1 = ds1_crop.ReadAsArray()
#     gt1_crop = ds1_crop.GetGeoTransform()
#     crs1 = ds1_crop.GetProjection()

#     ds2_crop = gdal.Translate(tmp_name, ds2, projWin=[left, top, right, bottom])
#     arr2 = ds2_crop.ReadAsArray()
#     gt2_crop = ds2_crop.GetGeoTransform()

#     # 掩码：只保留两张图都有值的区域
#     arr1_b1 = arr1[0] if arr1.ndim == 3 else arr1
#     arr2_b1 = arr2[0] if arr2.ndim == 3 else arr2
#     mask = (arr1_b1 != 0) & (arr2_b1 != 0)

#     if arr1.ndim == 3:
#         for b in range(arr1.shape[0]):
#             arr1[b][~mask] = 0
#         for b in range(arr2.shape[0]):
#             arr2[b][~mask] = 0
#     else:
#         arr1[~mask] = 0
#         arr2[~mask] = 0

#     # 清理
#     os.remove(tmp_name)
#     ds1 = ds2 = ds1_crop = ds2_crop = None

#     # ---------- 返回你要的字典格式 ----------
#     out1 = {
#         "image": arr1,
#         "data": arr1,
#         "transform": gt1_crop,
#         "crs": crs1,
#         "height": arr1.shape[-2],
#         "width": arr1.shape[-1]
#     }
#     out2 = {
#         "image": arr2,
#         "data": arr2,
#         "transform": gt2_crop,
#         "crs": crs1,
#         "height": arr2.shape[-2],
#         "width": arr2.shape[-1]
#     }

    
#     return out1, out2

def crop2(img1, img2, crop_points=None, max_size=32000, shrink_ratio=0.3):
    """
    裁剪两图公共区域
    
    Args:
        img1, img2: 输入影像（路径或字典）
        crop_points: 手动指定裁剪点（WGS84经纬度），为None时自动计算
        max_size: 最大输出尺寸（像素）
        shrink_ratio: 向内收缩比例（0.1 = 忽略上下左右各10%）
    """
    from osgeo import gdal, osr
    import numpy as np
    import tempfile
    import os

    # 打开图像（支持路径 或 字典）
    def open_img(img):
        if isinstance(img, str):
            return gdal.Open(img)
        else:
            driver = gdal.GetDriverByName('MEM')
            arr = img['image']
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            c, h, w = arr.shape
            ds = driver.Create('', w, h, c, gdal.GDT_UInt16)
            ds.SetGeoTransform(img['transform'])
            ds.SetProjection(img['crs'])
            for i in range(c):
                ds.GetRasterBand(i+1).WriteArray(arr[i])
            return ds

    ds1 = open_img(img1)
    ds2 = open_img(img2)

    gt1 = ds1.GetGeoTransform()
    gt2 = ds2.GetGeoTransform()
    resx1, resy1 = gt1[1], gt1[5]
    resx2, resy2 = gt2[1], gt2[5]

    # 坐标系
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(ds1.GetProjection())
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)

    # 计算裁剪范围
    if crop_points is not None:
        # 手动指定裁剪点
        transform = osr.CoordinateTransformation(dst_srs, src_srs)
        transformed_points = []
        for lon, lat in crop_points:
            x, y, _ = transform.TransformPoint(lon, lat, 0)
            transformed_points.append([x, y])
        
        xs = [p[0] for p in transformed_points]
        ys = [p[1] for p in transformed_points]
        left = min(xs)
        right = max(xs)
        top = max(ys)
        bottom = min(ys)
        
    else:
        # 自动计算公共区域，并收缩边缘
        left1, top1 = gt1[0], gt1[3]
        right1 = left1 + ds1.RasterXSize * resx1
        bottom1 = top1 + ds1.RasterYSize * resy1

        left2, top2 = gt2[0], gt2[3]
        right2 = left2 + ds2.RasterXSize * resx2
        bottom2 = top2 + ds2.RasterYSize * resy2

        # 计算交集（公共区域）
        left = max(left1, left2)
        right = min(right1, right2)
        top = min(top1, top2)
        bottom = max(bottom1, bottom2)
        
        # 检查是否有交集
        if left >= right or bottom >= top:
            raise ValueError("两图无公共区域")
        
        # 向内收缩（忽略边缘 shrink_ratio 比例）
        width = right - left
        height = top - bottom
        
        shrink_x = width * shrink_ratio
        shrink_y = height * shrink_ratio
        
        left += shrink_x
        right -= shrink_x
        top -= shrink_y
        bottom += shrink_y
        
        print(f"公共区域: ({left:.2f}, {bottom:.2f}, {right:.2f}, {top:.2f})")
        print(f"收缩后: 宽度={right-left:.2f}, 高度={top-bottom:.2f}")

    # 检查尺寸限制
    crop_width = int((right - left) / abs(resx1))
    crop_height = int((top - bottom) / abs(resy1))
    # print(f"预计裁剪尺寸: {crop_width} x {crop_height}")
    
    # 如果超过 max_size，进一步收缩
    if max(crop_width, crop_height) > max_size:
        scale = max_size / max(crop_width, crop_height)
        print(f"size too big，change ratio: {scale:.3f}")
        
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        
        new_width = (right - left) * scale
        new_height = (top - bottom) * scale
        
        left = center_x - new_width / 2
        right = center_x + new_width / 2
        top = center_y + new_height / 2
        bottom = center_y + new_height / 2
        
        crop_width = int(new_width / abs(resx1))
        crop_height = int(new_height / abs(resy1))
        # print(f"限制后尺寸: {crop_width} x {crop_height}")

    # ---------- 用 GDAL 稳定裁剪 ----------
    # 使用两个独立的临时文件
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp1:
        tmp_name1 = tmp1.name
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp2:
        tmp_name2 = tmp2.name

    try:
        ds1_crop = gdal.Translate(tmp_name1, ds1, projWin=[left, top, right, bottom])
        if ds1_crop is None:
            raise RuntimeError("gdal.Translate 影像1失败")
        arr1 = ds1_crop.ReadAsArray()
        gt1_crop = ds1_crop.GetGeoTransform()
        crs1 = ds1_crop.GetProjection()

        ds2_crop = gdal.Translate(tmp_name2, ds2, projWin=[left, top, right, bottom])
        if ds2_crop is None:
            raise RuntimeError("gdal.Translate 影像2失败")
        arr2 = ds2_crop.ReadAsArray()
        gt2_crop = ds2_crop.GetGeoTransform()

        # 掩码：只保留两张图都有值的区域
        arr1_b1 = arr1[0] if arr1.ndim == 3 else arr1
        arr2_b1 = arr2[0] if arr2.ndim == 3 else arr2
        mask = (arr1_b1 != 0) & (arr2_b1 != 0)

        if arr1.ndim == 3:
            for b in range(arr1.shape[0]):
                arr1[b][~mask] = 0
            for b in range(arr2.shape[0]):
                arr2[b][~mask] = 0
        else:
            arr1[~mask] = 0
            arr2[~mask] = 0

    finally:
        for f in [tmp_name1, tmp_name2]:
            if os.path.exists(f):
                os.remove(f)
        ds1 = ds2 = ds1_crop = ds2_crop = None

    # ---------- 返回字典格式 ----------
    out1 = {
        "image": arr1,
        "data": arr1,
        "transform": gt1_crop,
        "crs": crs1,
        "height": arr1.shape[-2],
        "width": arr1.shape[-1]
    }
    out2 = {
        "image": arr2,
        "data": arr2,
        "transform": gt2_crop,
        "crs": crs1,
        "height": arr2.shape[-2],
        "width": arr2.shape[-1]
    }

    return out1, out2










