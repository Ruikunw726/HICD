with open("F:/mambacd/home/HICD_v6/datasets/dataset_v6.py", "r", encoding="utf-8") as f:
    content = f.read()

old = """        result = {
            'img_t1': torch.from_numpy(pre_patch).float(),
            'img_t2': torch.from_numpy(post_patch).float(),
            'scene': patch_info['scene'],
            'base_name': patch_info['base_name']
        }"""

new = """        result = {
            'img_t1': torch.from_numpy(pre_patch).float(),
            'img_t2': torch.from_numpy(post_patch).float(),
            'scene': patch_info['scene'],
            'base_name': patch_info['base_name'],
            'patch_y': y,
            'patch_x': x,
            'img_h': H,
            'img_w': W
        }"""

content = content.replace(old, new)

with open("F:/mambacd/home/HICD_v6/datasets/dataset_v6.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done")
