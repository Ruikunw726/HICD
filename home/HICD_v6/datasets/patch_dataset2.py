with open("dataset_v6.py", "r", encoding="utf-8") as f:
    content = f.read()

# 替换__init__添加boxes_dir参数
content = content.replace(
    "    def __init__(self, data_dir, config, split='train', scenes=None, transform=None):",
    "    def __init__(self, data_dir, config, split='train', scenes=None, transform=None, boxes_dir=None):"
)

content = content.replace(
    """        self.data_dir = data_dir
        self.config = config
        self.split = split
        self.transform = transform""",
    """        self.data_dir = data_dir
        self.config = config
        self.split = split
        self.transform = transform
        self.boxes_dir = boxes_dir"""
)

# 替换__getitem__中的实例框提取逻辑
old_getitem = """        if label is not None:
            result['label'] = torch.from_numpy(label).long()
            
            # Generate instance bounding boxes from pixel labels
            instance_boxes = self._extract_instance_boxes(label)
            if instance_boxes is not None:
                result['instance_boxes'] = instance_boxes"""

new_getitem = """        if label is not None:
            result['label'] = torch.from_numpy(label).long()
            
            # Load pre-extracted instance bounding boxes
            if self.boxes_dir:
                box_path = os.path.join(
                    self.boxes_dir, sample['scene'], self.split, 'boxes',
                    sample['base_name'] + self.label_suffix + '.txt'
                )
                if os.path.exists(box_path):
                    boxes = self._load_boxes(box_path)
                    if boxes is not None:
                        result['instance_boxes'] = boxes"""

content = content.replace(old_getitem, new_getitem)

# 删除_extract_instance_boxes方法（不再需要），替换为_load_boxes
# 找到_extract_instance_boxes到collate_fn之间的内容
idx1 = content.find("    def _extract_instance_boxes")
idx2 = content.find("    @staticmethod\n    def collate_fn")

if idx1 > 0 and idx2 > 0:
    new_method = '''    def _load_boxes(self, box_path):
        """
        Load pre-extracted bounding boxes from file.
        
        File format: each line is "class_id cx cy w h" (normalized)
        
        Returns:
            torch.Tensor [N, 4] in (cx, cy, w, h) format, or None
        """
        boxes = []
        with open(box_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append([cx, cy, w, h])
        
        if len(boxes) == 0:
            return None
        
        return torch.tensor(boxes, dtype=torch.float32)
    
'''
    content = content[:idx1] + new_method + content[idx2:]

# 更新create_dataloader添加boxes_dir参数
content = content.replace(
    "def create_dataloader(data_dir, config, split='train', batch_size=4, \n                      num_workers=4, scenes=None, transform=None):",
    "def create_dataloader(data_dir, config, split='train', batch_size=4, \n                      num_workers=4, scenes=None, transform=None, boxes_dir=None):"
)

content = content.replace(
    """    dataset = HICDv6Dataset(
        data_dir=data_dir,
        config=config,
        split=split,
        scenes=scenes,
        transform=transform
    )""",
    """    dataset = HICDv6Dataset(
        data_dir=data_dir,
        config=config,
        split=split,
        scenes=scenes,
        transform=transform,
        boxes_dir=boxes_dir
    )"""
)

# 删除不再需要的scipy导入
content = content.replace("from scipy import ndimage\n", "")

with open("dataset_v6.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done")
