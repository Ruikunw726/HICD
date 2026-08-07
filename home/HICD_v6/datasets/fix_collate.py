with open("dataset_v6.py", "r", encoding="utf-8") as f:
    content = f.read()

old_collate = """    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return {}

        result = {
            'img_t1': torch.stack([b['img_t1'] for b in batch]),
            'img_t2': torch.stack([b['img_t2'] for b in batch]),
            'scene': [b['scene'] for b in batch],
            'base_name': [b['base_name'] for b in batch]
        }

        if 'label' in batch[0]:
            result['label'] = torch.stack([b['label'] for b in batch])

        result['instance_boxes'] = [b.get('instance_boxes') for b in batch]

        return result"""

new_collate = """    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return {}

        result = {
            'img_t1': torch.stack([b['img_t1'] for b in batch]),
            'img_t2': torch.stack([b['img_t2'] for b in batch]),
            'scene': [b['scene'] for b in batch],
            'base_name': [b['base_name'] for b in batch]
        }

        # Only stack labels if ALL samples have them
        if all('label' in b for b in batch):
            result['label'] = torch.stack([b['label'] for b in batch])

        result['instance_boxes'] = [b.get('instance_boxes') for b in batch]

        return result"""

content = content.replace(old_collate, new_collate)

with open("dataset_v6.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done")
