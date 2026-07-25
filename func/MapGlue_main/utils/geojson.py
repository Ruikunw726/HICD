import json
from rasterio.transform import xy   

def save_points_as_geojson(points, image_info, output_path, inliers=None):
    """将控制点保存为GeoJSON格式"""
    features = []
    
    # 获取图像的仿射变换参数
    transform = image_info.get('transform')
    crs = image_info.get('crs')
    
    # 确保CRS是字符串而不是CRS对象
    if hasattr(crs, 'to_string'):
        crs_str = crs.to_string()
    else:
        crs_str = str(crs) if crs is not None else None
    
    for i, (x, y) in enumerate(points):
    # 注意：points 里通常是 (col, row)，即 (x像素, y像素)
        if transform:
            geo_x, geo_y = xy(transform, int(y), int(x))  # 行=row=y, 列=col=x
        else:
            geo_x, geo_y = x, y
            
        # 创建点要素
        properties = {"id": i}
        
        # 添加内点标记（如果有）
        if inliers is not None and i < len(inliers):
            properties["inlier"] = bool(inliers[i])
            
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [geo_x, geo_y]
            },
            "properties": properties
        }
        features.append(feature)
    
    # 创建GeoJSON对象
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # 添加CRS信息（只保存字符串表示）
    if crs_str:
        geojson["crs"] = {
            "type": "name",
            "properties": {
                "name": crs_str
            }
        }
    
    # 保存文件
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"控制点已保存到: {output_path}")