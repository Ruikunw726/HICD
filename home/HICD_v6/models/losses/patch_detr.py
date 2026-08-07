import re

with open("detr_loss.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Fix HungarianMatcher to handle empty targets
old_matcher = """    @torch.no_grad()
    def forward(self, outputs, targets):
        \"\"\"
        Compute matching between predictions and targets
        
        Args:
            outputs: dict with 'pred_logits' and 'pred_boxes'
            targets: list of dicts with 'labels' and 'boxes'
            
        Returns:
            List of (pred_indices, target_indices) tuples
        \"\"\"
        bs, num_queries = outputs['pred_logits'].shape[:2]
        
        # Flatten to compute cost matrix
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # [bs*num_queries, num_classes+1]
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # [bs*num_queries, 4]
        
        # Concatenate all targets
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])"""

new_matcher = """    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_logits'].shape[:2]
        
        # Handle empty targets
        sizes = [len(v['boxes']) for v in targets]
        if sum(sizes) == 0:
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(bs)]
        
        # Flatten to compute cost matrix
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)
        
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])"""

content = content.replace(old_matcher, new_matcher)

# 2. Fix _loss_classification to handle empty indices
old_class = """    def _loss_classification(self, pred_logits, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            pred_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o"""

new_class = """    def _loss_classification(self, pred_logits, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        
        target_classes = torch.full(
            pred_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=pred_logits.device
        )
        
        if idx[0].numel() > 0:
            target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
            target_classes[idx] = target_classes_o"""

content = content.replace(old_class, new_class)

# 3. Fix _loss_boxes to handle empty indices
old_boxes = """    def _loss_boxes(self, pred_boxes, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)"""

new_boxes = """    def _loss_boxes(self, pred_boxes, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        
        if idx[0].numel() == 0:
            device = pred_boxes.device
            return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device, requires_grad=True)
        
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        if target_boxes.numel() == 0:
            device = pred_boxes.device
            return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device, requires_grad=True)"""

content = content.replace(old_boxes, new_boxes)

with open("detr_loss.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done")
