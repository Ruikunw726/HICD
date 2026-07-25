import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np


class InstanceDetectionLoss(nn.Module):
    """
    DETR 风格的匈牙利匹配损失。
    通过匈牙利算法将预测的 N 个 query 与 M 个 GT 实例最优匹配。
    """
    def __init__(self, num_classes, weight_bbox=5.0, weight_giou=2.0, weight_cls=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.weight_cls = weight_cls
        self.cls_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred_boxes, pred_logits, gt_boxes_list, gt_labels_list):
        """
        Args:
            pred_boxes: (B, N, 4) 预测 bbox (归一化 cx,cy,w,h)
            pred_logits: (B, N, num_classes+1) 分类 logits
            gt_boxes_list: list of (M_i, 4) 每张图的 GT bbox
            gt_labels_list: list of (M_i,) 每张图的 GT 类别
        Returns:
            total_loss, loss_dict
        """
        B = pred_boxes.shape[0]
        device = pred_boxes.device
        total_loss = torch.tensor(0.0, device=device)
        
        for b in range(B):
            gt_boxes = gt_boxes_list[b].to(device)
            gt_labels = gt_labels_list[b].to(device)
            M = gt_boxes.shape[0]
            
            if M == 0:
                # 没有 GT 实例时，只惩罚"无目标"类
                no_obj_target = torch.full(
                    (pred_logits.shape[1],), self.num_classes, 
                    dtype=torch.long, device=device
                )
                total_loss = total_loss + self.cls_loss(pred_logits[b], no_obj_target)
                continue
            
            # 匈牙利匹配
            cost_bbox = torch.cdist(pred_boxes[b], gt_boxes, p=1)  # (N, M)
            cost_giou = -self._generalized_box_iou(
                self._box_cxcywh_to_xyxy(pred_boxes[b]),
                self._box_cxcywh_to_xyxy(gt_boxes)
            )  # (N, M)
            
            # 分类代价: 取对应类别的负概率
            prob = pred_logits[b].softmax(-1)  # (N, num_classes+1)
            cost_cls = -prob[:, gt_labels]  # (N, M)
            
            cost_matrix = (
                self.weight_bbox * cost_bbox.detach().cpu().numpy() +
                self.weight_giou * cost_giou.detach().cpu().numpy() +
                self.weight_cls * cost_cls.detach().cpu().numpy()
            )
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 计算匹配后的损失
            matched_pred_boxes = pred_boxes[b][row_ind]
            matched_gt_boxes = gt_boxes[col_ind]
            matched_pred_logits = pred_logits[b][row_ind]
            matched_gt_labels = gt_labels[col_ind]
            
            # L1 bbox loss
            loss_bbox = F.l1_loss(matched_pred_boxes, matched_gt_boxes)
            
            # GIoU loss
            loss_giou = (1 - torch.diag(self._generalized_box_iou(
                self._box_cxcywh_to_xyxy(matched_pred_boxes),
                self._box_cxcywh_to_xyxy(matched_gt_boxes)
            ))).mean()
            
            # 分类 loss
            loss_cls = self.cls_loss(matched_pred_logits, matched_gt_labels)
            
            total_loss = total_loss + (
                loss_bbox * self.weight_bbox +
                loss_giou * self.weight_giou +
                loss_cls
            )
        
        return total_loss / B
    
    def _box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2]
        return torch.stack(b, dim=-1)
    
    def _generalized_box_iou(self, boxes1, boxes2):
        """计算 GIoU 矩阵"""
        inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
        inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
        inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
        inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
        
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        
        iou = inter / union.clamp(min=1e-6)
        
        enclose_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
        enclose_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
        enclose_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
        enclose_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
        enclose = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        return iou - (enclose - union) / enclose.clamp(min=1e-6)


class CombinedLoss(nn.Module):
    """
    总损失 = 像素级损失 + 实例级损失
    """
    def __init__(self, num_classes, weight_pixel=1.0, weight_instance=1.0):
        super().__init__()
        self.weight_pixel = weight_pixel
        self.weight_instance = weight_instance
        self.pixel_ce = nn.CrossEntropyLoss()
        self.instance_loss = InstanceDetectionLoss(num_classes)
    
    def forward(self, outputs, gt_pixel_change, gt_pixel_T1, gt_pixel_T2,
                gt_boxes_list, gt_labels_list):
        # 像素级损失
        loss_change = self.pixel_ce(outputs['pixel_change_map'], gt_pixel_change)
        loss_T1 = self.pixel_ce(outputs['pixel_T1_map'], gt_pixel_T1)
        loss_T2 = self.pixel_ce(outputs['pixel_T2_map'], gt_pixel_T2)
        loss_pixel = loss_change + loss_T1 + loss_T2
        
        # 实例级损失
        loss_instance = self.instance_loss(
            outputs['pred_boxes'], outputs['pred_logits'],
            gt_boxes_list, gt_labels_list
        )
        
        total = self.weight_pixel * loss_pixel + self.weight_instance * loss_instance
        
        return total, {
            'loss_change': loss_change.item(),
            'loss_T1': loss_T1.item(),
            'loss_T2': loss_T2.item(),
            'loss_instance': loss_instance.item(),
            'loss_total': total.item(),
        }
