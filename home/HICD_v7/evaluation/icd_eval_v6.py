"""
HICD v6 ICD Evaluation Protocol
Instance mAP + Semantic IoU/F1 + Full loss breakdown
"""
import torch
import numpy as np
from collections import defaultdict


def compute_iou_box(box1, box2):
    """Compute IoU between two boxes in [cx, cy, w, h] format."""
    b1 = _cxcywh_to_xyxy(box1)
    b2 = _cxcywh_to_xyxy(box2)
    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


def _cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]


class ICDEvaluatorV6:
    """
    ICD Evaluator for HICD v6
    - Instance branch: mAP@0.5, mAP@0.75, mAP@[0.5:0.95]
    - Semantic branch: IoU, F1, Accuracy
    - Loss breakdown: all components
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config
        branch_routing = dataset_config.get('branch_routing', {})
        self.instance_classes = branch_routing.get('instance', [])
        self.semantic_classes = branch_routing.get('semantic', [])
        self.target_names = dataset_config.get('target_names', [])
        self.state_names = dataset_config.get('state_names', [])
        self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.reset()

    def reset(self):
        self.instance_preds = []
        self.instance_gts = []
        self.semantic_target_preds = []
        self.semantic_target_gts = []
        self.semantic_state_preds = []
        self.semantic_state_gts = []
        self.loss_accum = defaultdict(float)
        self.loss_count = 0

    def update(self, predictions, targets, loss_dict=None):
        # Instance branch: collect pred boxes/logits + GT
        if 'instance_logits' in predictions and 'instance_labels' in targets:
            pred_logits = predictions['instance_logits'].detach()  # [B, Nq, C+1]
            pred_boxes = predictions['instance_boxes'].detach()    # [B, Nq, 4]
            inst_labels = targets['instance_labels']               # list of dicts

            for b in range(pred_logits.shape[0]):
                gt = inst_labels[b]
                gt_boxes = gt['boxes'].cpu().numpy()      # [M, 4]
                gt_labels = gt['labels'].cpu().numpy()     # [M]

                logits_b = pred_logits[b]  # [Nq, C+1]
                boxes_b = pred_boxes[b].cpu().numpy()  # [Nq, 4]

                # Get non-background predictions
                probs = logits_b.softmax(-1)  # [Nq, C+1]
                fg_probs = probs[:, :-1]      # [Nq, C]
                max_scores, pred_cls = fg_probs.max(-1)  # [Nq]

                # Filter by score threshold
                keep = max_scores > 0.1
                pred_boxes_keep = boxes_b[keep.cpu().numpy()]
                pred_cls_keep = pred_cls[keep].cpu().numpy()
                scores_keep = max_scores[keep].cpu().numpy()

                self.instance_preds.append({
                    'boxes': pred_boxes_keep,
                    'labels': pred_cls_keep,
                    'scores': scores_keep,
                })
                self.instance_gts.append({
                    'boxes': gt_boxes,
                    'labels': gt_labels,
                })

        # Semantic branch
        if 'target_logits' in predictions and 'target_gt' in targets:
            target_pred = predictions['target_logits'].argmax(dim=1)
            self.semantic_target_preds.append(target_pred.cpu())
            self.semantic_target_gts.append(targets['target_gt'].cpu())

        if 'state_logits' in predictions and 'state_gt' in targets:
            state_pred = predictions['state_logits'].argmax(dim=1)
            self.semantic_state_preds.append(state_pred.cpu())
            self.semantic_state_gts.append(targets['state_gt'].cpu())

        # Loss accumulation
        if loss_dict:
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    self.loss_accum[k] += v
            self.loss_count += 1

    def compute(self):
        results = {}

        # Instance mAP
        if self.instance_preds and self.instance_gts:
            results['instance_mAP'] = self._compute_ap_at_iou(0.5)
            results['instance_mAP@0.5'] = results['instance_mAP']
            results['instance_mAP@0.75'] = self._compute_ap_at_iou(0.75)
            aps = [self._compute_ap_at_iou(t) for t in self.iou_thresholds]
            results['instance_mAP@[0.5:0.95]'] = np.mean(aps) if aps else 0.0

            # Per-class AP
            results['instance_per_class'] = {}
            for cls_name in self.instance_classes:
                cls_idx = self.instance_classes.index(cls_name)
                cls_ap = self._compute_ap_at_iou_for_class(0.5, cls_idx)
                results['instance_per_class'][cls_name] = cls_ap
        else:
            results['instance_mAP'] = 0.0
            results['instance_mAP@0.5'] = 0.0
            results['instance_mAP@0.75'] = 0.0
            results['instance_mAP@[0.5:0.95]'] = 0.0
            results['instance_per_class'] = {}

        # Semantic target
        if self.semantic_target_preds:
            preds = torch.cat(self.semantic_target_preds, 0).numpy()
            gts = torch.cat(self.semantic_target_gts, 0).numpy()
            results['target_iou'] = self._compute_iou(preds, gts, len(self.semantic_classes))
            results['target_miou'] = float(np.mean(results['target_iou']))
            results['target_f1'] = self._compute_f1(preds, gts, len(self.semantic_classes))
            results['target_accuracy'] = float((preds == gts).mean())
            results['target_class_names'] = self.semantic_classes
        else:
            results['target_iou'] = []
            results['target_miou'] = 0.0
            results['target_f1'] = []
            results['target_accuracy'] = 0.0
            results['target_class_names'] = []

        # Semantic state
        if self.semantic_state_preds:
            preds = torch.cat(self.semantic_state_preds, 0).numpy()
            gts = torch.cat(self.semantic_state_gts, 0).numpy()
            results['state_iou'] = self._compute_iou(preds, gts, len(self.state_names))
            results['state_miou'] = float(np.mean(results['state_iou']))
            results['state_class_names'] = self.state_names
        else:
            results['state_iou'] = []
            results['state_miou'] = 0.0
            results['state_class_names'] = []

        # Loss averages
        results['loss_avg'] = {}
        for k, v in self.loss_accum.items():
            results['loss_avg'][k] = v / max(self.loss_count, 1)

        return results

    def _compute_ap_at_iou(self, iou_thresh):
        all_dets = []
        n_gt_total = 0
        for preds, gts in zip(self.instance_preds, self.instance_gts):
            gt_boxes = gts['boxes']
            gt_labels = gts['labels']
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)
            n_gt_total += len(gt_boxes)

            pred_boxes = preds['boxes']
            pred_labels = preds['labels']
            pred_scores = preds['scores']

            order = np.argsort(-pred_scores)
            for idx in order:
                box = pred_boxes[idx]
                cls = pred_labels[idx]
                score = pred_scores[idx]

                best_iou = 0.0
                best_gt = -1
                for gi, (gbox, gcls) in enumerate(zip(gt_boxes, gt_labels)):
                    if gt_matched[gi] or gcls != cls:
                        continue
                    iou = compute_iou_box(box, gbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gi

                if best_iou >= iou_thresh and best_gt >= 0:
                    all_dets.append((score, True))
                    gt_matched[best_gt] = True
                else:
                    all_dets.append((score, False))

        if n_gt_total == 0:
            return 0.0

        all_dets.sort(key=lambda x: -x[0])
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        for _, is_tp in all_dets:
            if is_tp:
                tp += 1
            else:
                fp += 1
            precisions.append(tp / (tp + fp))
            recalls.append(tp / n_gt_total)

        # 11-point interpolation AP
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            p_at_t = [p for p, r in zip(precisions, recalls) if r >= t]
            ap += max(p_at_t) if p_at_t else 0.0
        ap /= 11.0
        return ap

    def _compute_ap_at_iou_for_class(self, iou_thresh, cls_idx):
        all_dets = []
        n_gt_total = 0
        for preds, gts in zip(self.instance_preds, self.instance_gts):
            gt_boxes = gts['boxes']
            gt_labels = gts['labels']
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)

            mask_gt = gt_labels == cls_idx
            n_gt_total += mask_gt.sum()

            pred_boxes = preds['boxes']
            pred_labels = preds['labels']
            pred_scores = preds['scores']

            order = np.argsort(-pred_scores)
            for idx in order:
                if pred_labels[idx] != cls_idx:
                    continue
                box = pred_boxes[idx]
                score = pred_scores[idx]

                best_iou = 0.0
                best_gt = -1
                for gi, (gbox, gcls) in enumerate(zip(gt_boxes, gt_labels)):
                    if gt_matched[gi] or gcls != cls_idx:
                        continue
                    iou = compute_iou_box(box, gbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gi

                if best_iou >= iou_thresh and best_gt >= 0:
                    all_dets.append((score, True))
                    gt_matched[best_gt] = True
                else:
                    all_dets.append((score, False))

        if n_gt_total == 0:
            return 0.0

        all_dets.sort(key=lambda x: -x[0])
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        for _, is_tp in all_dets:
            if is_tp:
                tp += 1
            else:
                fp += 1
            precisions.append(tp / (tp + fp))
            recalls.append(tp / n_gt_total)

        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            p_at_t = [p for p, r in zip(precisions, recalls) if r >= t]
            ap += max(p_at_t) if p_at_t else 0.0
        ap /= 11.0
        return ap

    def _compute_iou(self, preds, gts, num_classes):
        ious = []
        for c in range(num_classes):
            p = (preds == c)
            g = (gts == c)
            inter = (p & g).sum()
            union = (p | g).sum()
            ious.append(float(inter / max(union, 1)))
        return ious

    def _compute_f1(self, preds, gts, num_classes):
        f1s = []
        for c in range(num_classes):
            tp = ((preds == c) & (gts == c)).sum()
            fp = ((preds == c) & (gts != c)).sum()
            fn = ((preds != c) & (gts == c)).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1s.append(float(2 * prec * rec / max(prec + rec, 1e-6)))
        return f1s

    def format_results(self, results):
        lines = []
        lines.append("=" * 60)
        lines.append("ICD Evaluation Results")
        lines.append("=" * 60)

        # Instance
        lines.append(f"\n[Instance Branch]")
        lines.append(f"  Classes: {self.instance_classes}")
        lines.append(f"  mAP@0.5:    {results.get('instance_mAP@0.5', 0):.4f}")
        lines.append(f"  mAP@0.75:   {results.get('instance_mAP@0.75', 0):.4f}")
        lines.append(f"  mAP@[.5:.95]: {results.get('instance_mAP@[0.5:0.95]', 0):.4f}")
        for cls_name, ap in results.get('instance_per_class', {}).items():
            lines.append(f"  {cls_name} AP@0.5: {ap:.4f}")

        # Loss breakdown
        loss_avg = results.get('loss_avg', {})
        if loss_avg:
            lines.append(f"\n[Loss Breakdown]")
            for k in sorted(loss_avg.keys()):
                lines.append(f"  {k}: {loss_avg[k]:.4f}")

        # Semantic target
        lines.append(f"\n[Semantic Branch - Target]")
        lines.append(f"  Classes: {results.get('target_class_names', [])}")
        for name, iou in zip(results.get('target_class_names', []), results.get('target_iou', [])):
            lines.append(f"  {name} IoU: {iou:.4f}")
        lines.append(f"  mIoU: {results.get('target_miou', 0):.4f}")
        for name, f1 in zip(results.get('target_class_names', []), results.get('target_f1', [])):
            lines.append(f"  {name} F1: {f1:.4f}")
        lines.append(f"  Accuracy: {results.get('target_accuracy', 0):.4f}")

        # State
        lines.append(f"\n[Semantic Branch - State]")
        lines.append(f"  Classes: {results.get('state_class_names', [])}")
        for name, iou in zip(results.get('state_class_names', []), results.get('state_iou', [])):
            lines.append(f"  {name} IoU: {iou:.4f}")
        lines.append(f"  mIoU: {results.get('state_miou', 0):.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)
