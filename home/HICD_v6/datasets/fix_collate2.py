with open("F:/mambacd/home/HICD_v6/datasets/dataset_v6.py", "r", encoding="utf-8") as f:
    content = f.read()

old = """        result = {
            'img_t1': torch.stack([b['img_t1'] for b in batch]),
            'img_t2': torch.stack([b['img_t2'] for b in batch]),
            'scene': [b['scene'] for b in batch],
            'base_name': [b['base_name'] for b in batch]
        }"""

new = """        result = {
            'img_t1': torch.stack([b['img_t1'] for b in batch]),
            'img_t2': torch.stack([b['img_t2'] for b in batch]),
            'scene': [b['scene'] for b in batch],
            'base_name': [b['base_name'] for b in batch],
            'patch_y': [b['patch_y'] for b in batch],
            'patch_x': [b['patch_x'] for b in batch],
            'img_h': batch[0]['img_h'],
            'img_w': batch[0]['img_w']
        }"""

content = content.replace(old, new)

with open("F:/mambacd/home/HICD_v6/datasets/dataset_v6.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done")
