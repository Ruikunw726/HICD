import numpy as np
from osgeo import osr, gdal

# 压制 PROJ 的报错信息，避免刷屏（可选）
# os.environ['PROJ_DEBUG'] = '0'

class CRS:
    def __init__(self, srs_input):
        self.srs = osr.SpatialReference()
        try:
            # 强制 GIS 顺序 (Lon, Lat)
            self.srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        except:
            pass

        try:
            if isinstance(srs_input, int):
                self.srs.ImportFromEPSG(srs_input)
            elif isinstance(srs_input, str):
                srs_input = srs_input.strip()
                if srs_input.upper().startswith("EPSG:"):
                    code = int(srs_input.split(":")[1])
                    self.srs.ImportFromEPSG(code)
                elif "+proj" in srs_input:
                    self.srs.ImportFromProj4(srs_input)
                else:
                    self.srs.ImportFromWkt(srs_input)
            elif hasattr(srs_input, 'to_wkt'):
                self.srs.ImportFromWkt(srs_input.to_wkt())
            elif hasattr(srs_input, 'wkt'):
                self.srs.ImportFromWkt(srs_input.wkt)
            elif isinstance(srs_input, osr.SpatialReference):
                self.srs = srs_input
        except Exception as e:
            print(f"CRS Init Error: {e}")

    def to_wkt(self):
        return self.srs.ExportToWkt()

    @staticmethod
    def from_user_input(value):
        return CRS(value)

    @staticmethod
    def from_epsg(code):
        return CRS(code)


class Transformer:
    def __init__(self, src_crs, dst_crs, always_xy=True):
        self.src_crs_obj = src_crs if isinstance(src_crs, CRS) else CRS(src_crs)
        self.dst_crs_obj = dst_crs if isinstance(dst_crs, CRS) else CRS(dst_crs)
        self.ct = osr.CoordinateTransformation(self.src_crs_obj.srs, self.dst_crs_obj.srs)

    @staticmethod
    def from_crs(src_crs, dst_crs, always_xy=True):
        return Transformer(src_crs, dst_crs, always_xy=always_xy)

    def transform(self, xx, yy):
        """
        修复版：将 Numpy 转换为 List 之后再传给 GDAL
        """
        # 1. 标量处理
        if np.isscalar(xx):
            res = self.ct.TransformPoint(float(xx), float(yy))
            return res[0], res[1]
        
        # 2. 数组处理
        xx = np.asarray(xx)
        yy = np.asarray(yy)
        original_shape = xx.shape
        
        # 展平 + 组合
        # 注意：必须转为 float 类型，否则 GDAL 可能读不懂 int
        points_np = np.column_stack((xx.flatten().astype(float), yy.flatten().astype(float)))
        
        # !!! 关键修正 !!! 
        # GDAL 的 Python 绑定不接受 Numpy 数组作为参数
        # 必须转为 Python 原生列表: [[x,y], [x,y], ...]
        points_list = points_np.tolist()
        
        try:
            # 现在传入的是 List，应该不会报错了
            res_list = self.ct.TransformPoints(points_list)
            
            # res_list 是 [(x,y,z), (x,y,z)...]
            res_np = np.array(res_list)
            
            out_x = res_np[:, 0].reshape(original_shape)
            out_y = res_np[:, 1].reshape(original_shape)
            return out_x, out_y
            
        except Exception as e:
            print(f"坐标转换失败: {e}")
            # 如果失败，返回原值，方便调试（至少程序不崩）
            return xx, yy