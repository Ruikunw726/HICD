import re

with open("dataset_v6.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add scipy import
content = content.replace(
    "from PIL import Image",
    "from PIL import Image\nfrom scipy import ndimage"
)

# 2. Add instance_boxes generation in __getitem__
old_getitem_end = """        if label is not None:
            result['label'] = torch.from_numpy(label).long()
        
        return result"""

new_getitem_end = """        if label is not None:
            result['label'] = torch.from_numpy(label).long()
            
            # Generate instance bounding boxes from pixel labels
            instance_boxes = self._extract_instance_boxes(label)
            if instance_boxes is not None:
                result['instance_boxes'] = instance_boxes
        
        return result"""

content = content.replace(old_getitem_end, new_getitem_end)

# 3. Add _extract_instance_boxes and collate_fn before __main__
extract_method = '''
    def _extract_instance_boxes(self, label):
        """
        Extract instance bounding boxes from pixel-level labels.
        Uses connected component analysis to find individual instances.
        
        Args:
            label: [H, W] pixel-level label (0=background/damage, >0=damage levels)
            
        Returns:
            boxes: [N, 4] bounding boxes in (cx, cy, w, h) normalized format
        """
        # Any non-zero pixel is a building instance
        binary_mask = (label > 0).astype('uint8')
        
        if binary_mask.sum() == 0:
            return None
        
        # Label connected components
        labeled, num_instances = ndimage.label(binary_mask)
        
        if num_instances == 0:
            return None
        
        boxes = []
        h, w = label.shape
        
        for i in range(1, num_instances + 1):
            instance_mask = (labeled == i)
            y_coords, x_coords = np.where(instance_mask)
            
            # Filter tiny instances (noise)
            if len(y_coords) < 10:
                continue
            
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            # Convert to normalized (cx, cy, w, h)
            cx = (x_min + x_max) / 2.0 / w
            cy = (y_min + y_max) / 2.0 / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            
            boxes.append([cx, cy, bw, bh])
        
        if len(boxes) == 0:
            return None
        
        return torch.tensor(boxes, dtype=torch.float32)
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate to handle variable number of instances"""
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
        
        # Variable-length instance boxes
        result['instance_boxes'] = [b.get('instance_boxes') for b in batch]
        
        return result

'''

content = content.replace(
    "\\nif __name__ == '__main__':",
    extract_method + "\\nif __name__ == '__main__':"
)

with open("dataset_v6.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done")
