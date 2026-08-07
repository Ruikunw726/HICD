with open("/mnt/f/mambacd/home/HICD_v6/datasets/dataset_v6.py", "r") as f:
    content = f.read()

load_boxes = '''
    def _load_boxes(self, box_path):
        boxes = []
        with open(box_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append([cx, cy, w, h])
        if len(boxes) == 0:
            return None
        return torch.tensor(boxes, dtype=torch.float32)

'''

if "def _load_boxes" not in content:
    content = content.replace(
        "    @staticmethod\n    def collate_fn",
        load_boxes + "    @staticmethod\n    def collate_fn"
    )

with open("/mnt/f/mambacd/home/HICD_v6/datasets/dataset_v6.py", "w") as f:
    f.write(content)

print("Done")
