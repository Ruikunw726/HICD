#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Instance-level Change Detection Evaluation (ICD-Eval)

Core idea: "Did the model detect that this target changed?"
  - Instance methods: match predicted boxes to GT boxes via IoU
  - Pixel methods: for each GT instance, check if enough pixels inside
    are predicted as changed (IoU-based)
  - Large objects: split into sub-regions for fair partial-hit evaluation

Metrics: ICD-Precision, ICD-Recall, ICD-F1, ICD-mAP

Usage:
    # Instance model
    python eval_unified.py --mode instance --checkpoint outputs/best.pth ...

    # Pixel predictions
    python eval_unified.py --mode pixel --pred_dir /path/to/preds ...
"""
import warnings
warnings.filterwarnings('ignore')
import sys, os, json, argparse, time
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from MambaCD.changedetection.configs.config import get_config
from MambaCD.changedetection.datasets.imutils import normalize_img
from MambaCD.changedetection.models.HierarchicalSCD_Instance import HierarchicalSCDInstance
from MambaCD.changedetection.models.class_mapping import TARGET_NAMES, STATE_NAMES, NUM_TARGETS, NUM_STATES
from osgeo import gdal
gdal.UseExceptions()



# =====================================================================
# train_id mapping from classes.csv
# (target_idx, state_idx) -> train_id
# target_idx: 0-15 (TARGET_NAMES), state_idx: 0-5 (STATE_NAMES)
# =====================================================================
TARGET_STATE_TO_TRAIN_ID = {
    (0,0):1, (0,1):2,
    (1,0):3, (1,1):4, (1,2):5, (1,3):6, (1,4):7,
    (2,0):8, (2,1):9, (2,2):10, (2,3):11, (2,4):12,
    (3,0):13, (3,1):14, (3,2):15, (3,3):16, (3,4):17,
    (4,0):18, (4,1):19, (4,2):20, (4,3):21, (4,4):22,
    (5,0):23, (5,1):24, (5,2):25, (5,3):26, (5,4):27,
    (6,0):28, (6,1):29, (6,2):30, (6,3):31, (6,4):32,
    (7,0):33, (7,1):34, (7,2):35, (7,3):36, (7,4):37,
    (8,0):38, (8,1):39, (8,2):40, (8,3):41, (8,4):42,
    (9,0):43, (9,1):44, (9,2):45, (9,3):46, (9,4):47,
    (10,0):48, (10,1):49, (10,2):50, (10,3):51, (10,4):52,
    (11,0):53, (11,1):54, (11,2):55, (11,3):56,
    (12,0):57, (12,1):58, (12,2):59, (12,3):60, (12,5):61,
    (13,0):62, (13,1):63, (13,2):64, (13,3):65, (13,5):66,
    (14,0):67,
    (15,0):68,
}

TRAIN_ID_TO_TARGET_STATE = {v: k for k, v in TARGET_STATE_TO_TRAIN_ID.items()}
def load_gt_instances(json_path, split="test"):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt = {}
    for key, val in data.items():
        if not key.startswith(f"{split}/"):
            continue
        fname = key.split("/", 1)[1]
        instances = val.get('instances', [])
        if not instances:
            continue
        gt[fname] = {
            'boxes': [inst['bbox'] for inst in instances],
            'targets': [inst['target_idx'] for inst in instances],
            'states': [inst['state_idx'] for inst in instances],
            'areas': [inst.get('area', 0) for inst in instances],
        }
    return gt


def box_to_mask(box, H, W):
    cx, cy, bw, bh = box
    x1, y1 = max(0, int((cx-bw/2)*W)), max(0, int((cy-bh/2)*H))
    x2, y2 = min(W, int((cx+bw/2)*W)), min(H, int((cy+bh/2)*H))
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def split_large_instance(box, H, W, area_thresh=5000, n_segments=4):
    cx, cy, bw, bh = box
    pixel_area = (bw*W) * (bh*H)
    if pixel_area <= area_thresh:
        return [box]
    if bw*W >= bh*H:
        seg_w = bw / n_segments
        return [[cx - bw/2 + seg_w*(i+0.5), cy, seg_w, bh] for i in range(n_segments)]
    else:
        seg_h = bh / n_segments
        return [[cx, cy - bh/2 + seg_h*(i+0.5), bw, seg_h] for i in range(n_segments)]


def mask_iou(m1, m2):
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return inter / max(union, 1)


def box_iou_xyxy(b1, b2):
    ix1, iy1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    ix2, iy2 = min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / max(a1+a2-inter, 1)


def cxcywh_to_xyxy(box):
    cx,cy,w,h = box
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]

# =====================================================================
# Core Evaluator
# =====================================================================
class ICDEvaluator:
    def __init__(self, iou_thresholds=None, large_area_thresh=5000, n_sub_regions=4):
        self.iou_thresholds = iou_thresholds or [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        self.large_area_thresh = large_area_thresh
        self.n_sub_regions = n_sub_regions
        self.change_state_ids = [1,2,3,4,5]  # Not NoChange

    def evaluate_instance(self, predictions, gt_data):
        all_gt, all_pred = [], []
        for fname in gt_data:
            if fname not in predictions:
                continue
            pred, gt = predictions[fname], gt_data[fname]
            for i in range(len(gt['boxes'])):
                all_gt.append({'fname':fname,'idx':i,'box':gt['boxes'][i],
                    'target':gt['targets'][i],'state':gt['states'][i],
                    'is_changed': gt['states'][i] in self.change_state_ids,
                    'matched':False})
            for j in range(pred['boxes'].shape[0]):
                all_pred.append({'fname':fname,'box':pred['boxes'][j].tolist(),
                    'target':pred['targets'][j].item() if torch.is_tensor(pred['targets'][j]) else pred['targets'][j],
                    'state':pred['states'][j].item() if torch.is_tensor(pred['states'][j]) else pred['states'][j],
                    'score':pred['scores'][j].item() if torch.is_tensor(pred['scores'][j]) else pred['scores'][j]})
        all_pred.sort(key=lambda x: -x['score'])
        results = {}
        for iou_t in self.iou_thresholds:
            for g in all_gt:
                g['matched'] = False
            tp, fp = 0, 0
            for pred in all_pred:
                best_iou, best_gt = 0, None
                for g in all_gt:
                    if g['fname']!=pred['fname'] or g['matched']:
                        continue
                    iou = box_iou_xyxy(cxcywh_to_xyxy(pred['box']), cxcywh_to_xyxy(g['box']))
                    if iou > best_iou:
                        best_iou, best_gt = iou, g
                if best_iou >= iou_t and best_gt is not None and best_gt['is_changed']:
                    tp += 1
                    best_gt['matched'] = True
                else:
                    fp += 1
            n_gt_changed = sum(1 for g in all_gt if g['is_changed'])
            p = tp/max(tp+fp,1)
            r = tp/max(n_gt_changed,1)
            f1 = 2*p*r/max(p+r,1e-6)
            results[iou_t] = {'tp':tp,'fp':fp,'n_gt':n_gt_changed,
                              'precision':p,'recall':r,'f1':f1}
        aps = [results[t]['f1'] for t in self.iou_thresholds]
        results['mAP'] = np.mean(aps)
        results['mAP_50'] = results.get(0.5,{}).get('f1',0)
        results['mAP_75'] = results.get(0.75,{}).get('f1',0)
        return results

    def evaluate_pixel(self, pred_dir, gt_data, split="test"):
        res_per_t = {t:{'tp':0,'fp':0,'fn':0,'n_gt':0} for t in self.iou_thresholds}
        for fname, gt in tqdm(gt_data.items(), desc="Pixel eval"):
            pred_path = self._find_pred(pred_dir, fname)
            if not pred_path:
                continue
            pred_map = self._load_map(pred_path)
            if pred_map is None:
                continue
            H, W = pred_map.shape[:2]
            for i in range(len(gt['boxes'])):
                is_changed = gt['states'][i] in self.change_state_ids
                subs = split_large_instance(gt['boxes'][i], H, W, self.large_area_thresh, self.n_sub_regions)
                for sub in subs:
                    gt_mask = box_to_mask(sub, H, W)
                    pred_mask = (pred_map>0).astype(np.uint8) if pred_map.ndim==2 else pred_map.astype(np.uint8)
                    iou = mask_iou(gt_mask, pred_mask)
                    for t in self.iou_thresholds:
                        if is_changed:
                            res_per_t[t]['n_gt'] += 1
                            if iou >= t:
                                res_per_t[t]['tp'] += 1
                            else:
                                res_per_t[t]['fn'] += 1
                        elif iou >= t:
                            res_per_t[t]['fp'] += 1
        results = {}
        for t in self.iou_thresholds:
            r = res_per_t[t]
            p = r['tp']/max(r['tp']+r['fp'],1)
            rec = r['tp']/max(r['n_gt'],1)
            f1 = 2*p*rec/max(p+rec,1e-6)
            results[t] = {'tp':r['tp'],'fp':r['fp'],'fn':r['fn'],'n_gt':r['n_gt'],
                          'precision':p,'recall':rec,'f1':f1}
        aps = [results[t]['f1'] for t in self.iou_thresholds]
        results['mAP'] = np.mean(aps)
        results['mAP_50'] = results.get(0.5,{}).get('f1',0)
        results['mAP_75'] = results.get(0.75,{}).get('f1',0)
        return results

    def _find_pred(self, pred_dir, fname):
        stem = os.path.splitext(fname)[0]
        for suffix in ['_pred.tif','_change.tif','.png','_pred.png']:
            p = os.path.join(pred_dir, stem+suffix)
            if os.path.exists(p):
                return p
        p = os.path.join(pred_dir, fname)
        return p if os.path.exists(p) else None

    def _load_map(self, path):
        try:
            if path.endswith('.tif') or path.endswith('.tiff'):
                ds = gdal.Open(path)
                if ds is None: return None
                arr = ds.ReadAsArray(); ds = None
                return arr[0] if arr.ndim==3 else arr
            else:
                from PIL import Image
                arr = np.array(Image.open(path))
                return arr[:,:,0] if arr.ndim==3 else arr
        except:
            return None

    def evaluate_scd(self, pred_dir, gt_data, pred_format="change_map",
                     pre_dir=None, post_dir=None):
        """Evaluate SCD (Semantic Change Detection) predictions.
        
        SCD outputs per-pixel semantic labels. We convert to instance-level:
        For each GT instance bbox, check if predicted pixels match target type.
        
        Args:
            pred_dir: directory with prediction maps
            gt_data: GT instances
            pred_format: "change_map" (single map with change classes)
                        or "pre_post" (two separate semantic maps)
            pre_dir, post_dir: directories for pre/post maps (if pred_format="pre_post")
        """
        res_per_t = {t:{'tp':0,'fp':0,'fn':0,'n_gt':0} for t in self.iou_thresholds}
        
        for fname, gt in tqdm(gt_data.items(), desc="SCD eval"):
            if pred_format == "change_map":
                pred_path = self._find_pred(pred_dir, fname)
                if not pred_path:
                    continue
                pred_map = self._load_map(pred_path)
                if pred_map is None:
                    continue
            else:
                # pre_post format
                pre_path = self._find_pred(pre_dir, fname)
                post_path = self._find_pred(post_dir, fname)
                if not pre_path or not post_path:
                    continue
                pre_map = self._load_map(pre_path)
                post_map = self._load_map(post_path)
                if pre_map is None or post_map is None:
                    continue
                # Change = pixels where pre != post AND post != 0 (background)
                pred_map = np.zeros_like(post_map)
                changed = (pre_map != post_map) & (post_map > 0)
                pred_map[changed] = post_map[changed]
            
            H, W = pred_map.shape[:2]
            
            for i in range(len(gt['boxes'])):
                is_changed = gt['states'][i] in self.change_state_ids
                gt_target = gt['targets'][i]  # 0-15
                
                subs = split_large_instance(gt['boxes'][i], H, W,
                    self.large_area_thresh, self.n_sub_regions)
                
                for sub in subs:
                    gt_mask = box_to_mask(sub, H, W)
                    
                    # For change_map: pixels with matching target class
                    # Target classes in SCD maps are typically 1-indexed (0=background)
                    pred_class_mask = (pred_map == (gt_target + 1)).astype(np.uint8)
                    
                    iou = mask_iou(gt_mask, pred_class_mask)
                    
                    for t in self.iou_thresholds:
                        if is_changed:
                            res_per_t[t]['n_gt'] += 1
                            if iou >= t:
                                res_per_t[t]['tp'] += 1
                            else:
                                res_per_t[t]['fn'] += 1
                        elif iou >= t:
                            res_per_t[t]['fp'] += 1
        
        results = {}
        for t in self.iou_thresholds:
            r = res_per_t[t]
            p = r['tp']/max(r['tp']+r['fp'],1)
            rec = r['tp']/max(r['n_gt'],1)
            f1 = 2*p*rec/max(p+rec,1e-6)
            results[t] = {'tp':r['tp'],'fp':r['fp'],'fn':r['fn'],'n_gt':r['n_gt'],
                          'precision':p,'recall':rec,'f1':f1}
        aps = [results[t]['f1'] for t in self.iou_thresholds]
        results['mAP'] = np.mean(aps)
        results['mAP_50'] = results.get(0.5,{}).get('f1',0)
        results['mAP_75'] = results.get(0.75,{}).get('f1',0)
        return results


# =====================================================================
# Per-class Evaluation
# =====================================================================
def evaluate_per_class(predictions, gt_data, iou_threshold=0.3):
    change_ids = [1,2,3,4,5]
    tgt_stats = defaultdict(lambda: {'tp':0,'fp':0,'fn':0})
    st_stats = defaultdict(lambda: {'tp':0,'fp':0,'fn':0})
    for fname in gt_data:
        if fname not in predictions:
            continue
        pred, gt = predictions[fname], gt_data[fname]
        n_p, n_g = pred['boxes'].shape[0], len(gt['boxes'])
        if n_g==0 or n_p==0:
            for i in range(n_g):
                if gt['states'][i] in change_ids:
                    tgt_stats[gt['targets'][i]]['fn'] += 1
                    st_stats[gt['states'][i]]['fn'] += 1
            continue
        iou_m = np.zeros((n_p, n_g))
        for pi in range(n_p):
            for gi in range(n_g):
                iou_m[pi,gi] = box_iou_xyxy(cxcywh_to_xyxy(pred['boxes'][pi].tolist()),
                                             cxcywh_to_xyxy(gt['boxes'][gi]))
        matched = set()
        for pi in np.argsort(-pred['scores'].numpy()):
            best_iou, best_gi = 0, -1
            for gi in range(n_g):
                if gi not in matched and iou_m[pi,gi]>best_iou:
                    best_iou, best_gi = iou_m[pi,gi], gi
            pt = pred['targets'][pi].item() if torch.is_tensor(pred['targets'][pi]) else pred['targets'][pi]
            ps = pred['states'][pi].item() if torch.is_tensor(pred['states'][pi]) else pred['states'][pi]
            if best_iou>=iou_threshold and best_gi>=0:
                if gt['states'][best_gi] in change_ids:
                    matched.add(best_gi)
                    if pt==gt['targets'][best_gi]:
                        tgt_stats[gt['targets'][best_gi]]['tp']+=1
                    else:
                        tgt_stats[gt['targets'][best_gi]]['fn']+=1; tgt_stats[pt]['fp']+=1
                    if ps==gt['states'][best_gi]:
                        st_stats[gt['states'][best_gi]]['tp']+=1
                    else:
                        st_stats[gt['states'][best_gi]]['fn']+=1; st_stats[ps]['fp']+=1
                else:
                    tgt_stats[pt]['fp']+=1; st_stats[ps]['fp']+=1
            else:
                tgt_stats[pt]['fp']+=1; st_stats[ps]['fp']+=1
        for gi in range(n_g):
            if gi not in matched and gt['states'][gi] in change_ids:
                tgt_stats[gt['targets'][gi]]['fn']+=1; st_stats[gt['states'][gi]]['fn']+=1
    def _f1(stats):
        res = {}
        for cid, s in stats.items():
            p=s['tp']/max(s['tp']+s['fp'],1); r=s['tp']/max(s['tp']+s['fn'],1)
            res[cid] = {'P':p,'R':r,'F1':2*p*r/max(p+r,1e-6),'support':s['tp']+s['fn']}
        return res
    tgt_res = {TARGET_NAMES[k]:v for k,v in _f1(tgt_stats).items() if k<len(TARGET_NAMES)}
    st_res = {STATE_NAMES[k]:v for k,v in _f1(st_stats).items() if k<len(STATE_NAMES)}
    return tgt_res, st_res


# =====================================================================
# Dataset & Inference
# =====================================================================
class InstanceTestDataset(Dataset):
    def __init__(self, dataset_path, instances_dict, crop_size=512):
        self.dataset_path = dataset_path
        self.crop_size = crop_size
        self.instances_dict = instances_dict
        train_dir = os.path.join(dataset_path, "train", "image")
        self.flat_structure = not os.path.isdir(os.path.join(train_dir, "pre"))
        self.samples = [(k.split("/",1)[0], k.split("/",1)[1]) for k in instances_dict.keys()]
    def __len__(self): return len(self.samples)
    def __getitem__(self, index):
        split, fname = self.samples[index]
        stem = os.path.splitext(fname)[0].replace('_target','')
        if self.flat_structure:
            pre_path = os.path.join(self.dataset_path,split,"image",stem+"_pre_war.tif")
            post_path = os.path.join(self.dataset_path,split,"image",stem+"_post_war.tif")
        else:
            pre_path = os.path.join(self.dataset_path,split,"image","pre",fname)
            post_path = os.path.join(self.dataset_path,split,"image","post",fname)
        pre = self._read(pre_path).astype(np.float32)
        post = self._read(post_path).astype(np.float32)
        H,W,C = pre.shape
        if H<self.crop_size or W<self.crop_size:
            ph,pw = max(self.crop_size-H,0), max(self.crop_size-W,0)
            pre = np.pad(pre,((0,ph),(0,pw),(0,0)),'reflect')
            post = np.pad(post,((0,ph),(0,pw),(0,0)),'reflect')
        if H>self.crop_size or W>self.crop_size:
            y,x = (H-self.crop_size)//2, (W-self.crop_size)//2
            pre,post = pre[y:y+self.crop_size,x:x+self.crop_size], post[y:y+self.crop_size,x:x+self.crop_size]
        pre = np.transpose(normalize_img(pre),(2,0,1)).astype(np.float32)
        post = np.transpose(normalize_img(post),(2,0,1)).astype(np.float32)
        return {'pre_img':torch.from_numpy(pre),'post_img':torch.from_numpy(post),
                'filename':fname,'orig_hw':torch.tensor([H,W])}
    def _read(self, path):
        ds = gdal.Open(path); arr = ds.ReadAsArray(); ds = None
        if arr.ndim==2: arr=arr[:,:,np.newaxis]
        if arr.shape[0]<=4: arr=np.transpose(arr,(1,2,0))
        return arr[:,:,:3] if arr.shape[2]>3 else arr


def instance_collate(batch):
    return {'pre_img':torch.stack([b['pre_img'] for b in batch]),
            'post_img':torch.stack([b['post_img'] for b in batch]),
            'filename':[b['filename'] for b in batch],
            'orig_hw':torch.stack([b['orig_hw'] for b in batch])}


def run_inference(model, loader, device, score_thresh=0.3):
    model.eval(); preds = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            pre,post = batch['pre_img'].to(device), batch['post_img'].to(device)
            fnames = batch['filename']
            out = model(pre, post)
            pb, pt, ps = out['pred_boxes'].cpu(), out['pred_target'].cpu(), out['pred_state'].cpu()
            for b in range(pre.shape[0]):
                probs = torch.softmax(pt[b], dim=-1)
                scores, targets = probs.max(dim=-1)
                states = ps[b].argmax(dim=-1)
                mask = scores > score_thresh
                preds[fnames[b]] = {'boxes':pb[b][mask],'targets':targets[mask],
                                    'states':states[mask],'scores':scores[mask]}
    return preds


# =====================================================================
# Print & Main
# =====================================================================
def print_icd(results, title="ICD"):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    print(f"  {'IoU':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"  {'-'*45}")
    for t in sorted([k for k in results if isinstance(k, float)]):
        r = results[t]
        print(f"  {t:>5.2f} {r['tp']:>5} {r['fp']:>5} {r.get('fn',0):>5} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}")
    print(f"\n  mAP      = {results.get('mAP',0):.4f}")
    print(f"  mAP@0.5  = {results.get('mAP_50',0):.4f}")
    print(f"  mAP@0.75 = {results.get('mAP_75',0):.4f}")


def print_per_class(tgt_res, st_res):
    print(f"\n{'='*60}\n  Per-class (IoU=0.3)\n{'='*60}")
    print(f"\n  {'Target':>20} {'P':>7} {'R':>7} {'F1':>7} {'N':>5}")
    print(f"  {'-'*46}")
    for n,r in sorted(tgt_res.items(), key=lambda x:-x[1]['F1']):
        print(f"  {n:>20} {r['P']:>7.4f} {r['R']:>7.4f} {r['F1']:>7.4f} {r['support']:>5}")
    print(f"\n  {'State':>20} {'P':>7} {'R':>7} {'F1':>7} {'N':>5}")
    print(f"  {'-'*46}")
    for n,r in sorted(st_res.items(), key=lambda x:-x[1]['F1']):
        print(f"  {n:>20} {r['P']:>7.4f} {r['R']:>7.4f} {r['F1']:>7.4f} {r['support']:>5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified ICD Evaluation")
    parser.add_argument("--mode", type=str, default="instance", choices=["instance","pixel","scd"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    parser.add_argument("--pred_format", type=str, default="change_map",
                        choices=["change_map","pre_post"],
                        help="SCD prediction format")
    parser.add_argument("--pre_dir", type=str, default=None, help="Pre-change map dir (for pre_post)")
    parser.add_argument("--post_dir", type=str, default=None, help="Post-change map dir (for pre_post)")
    parser.add_argument("--data_dir", type=str, default="MambaCD/0617final")
    parser.add_argument("--scenes", type=str, default="Airports,Ports,Urban-Rural Areas")
    parser.add_argument("--cfg", type=str, default="MambaCD/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--num_queries", type=int, default=17)
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--large_area_thresh", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--clip_weights_path", type=str, default="MambaCD/weights/open_clip_pytorch_model.bin")
    parser.add_argument("--pretrained_weight_path", type=str, default="MambaCD/weights/vssmtiny_dp01_ckpt_epoch_292.pth")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load GT
    print("\nLoading GT...")
    all_gt = {}
    for scene in args.scenes.split(","):
        sd = os.path.join(args.data_dir, scene.strip())
        jp = os.path.join(sd, "instances.json")
        if not os.path.exists(jp): continue
        gt = load_gt_instances(jp, "test")
        print(f"  {scene.strip()}: {len(gt)} images")
        all_gt.update(gt)
    print(f"Total: {len(all_gt)} test images")

    evaluator = ICDEvaluator(large_area_thresh=args.large_area_thresh)

    if args.mode == "instance":
        if not args.checkpoint:
            print("ERROR: --checkpoint required"); sys.exit(1)
        test_datasets = []
        for scene in args.scenes.split(","):
            sd = os.path.join(args.data_dir, scene.strip())
            with open(os.path.join(sd,"instances.json"),'r') as f:
                inst = json.load(f)
            ti = {k:v for k,v in inst.items() if k.startswith("test/")}
            if ti: test_datasets.append(InstanceTestDataset(sd, ti, args.crop_size))
        ds = ConcatDataset(test_datasets) if len(test_datasets)>1 else test_datasets[0]
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=instance_collate, pin_memory=True)
        # Build model
        cfg = get_config(args); cfg.defrost(); vssm = cfg.MODEL.VSSM
        cfg_d = {'norm_layer':vssm.NORM_LAYER,'ssm_act_layer':vssm.SSM_ACT_LAYER,
            'mlp_act_layer':vssm.MLP_ACT_LAYER,'ssm_d_state':vssm.SSM_D_STATE,
            'ssm_ratio':vssm.SSM_RATIO,'ssm_dt_rank':vssm.SSM_DT_RANK,
            'ssm_conv':vssm.SSM_CONV,'ssm_conv_bias':vssm.SSM_CONV_BIAS,
            'ssm_drop_rate':vssm.SSM_DROP_RATE,'ssm_init':vssm.SSM_INIT,
            'forward_type':vssm.SSM_FORWARDTYPE,'mlp_ratio':vssm.MLP_RATIO,
            'mlp_drop_rate':vssm.MLP_DROP_RATE,'gmlp':vssm.GMLP,
            'use_checkpoint':cfg.TRAIN.USE_CHECKPOINT,'drop_path_rate':cfg.MODEL.DROP_PATH_RATE,
            'patch_size':vssm.PATCH_SIZE,'in_chans':vssm.IN_CHANS,
            'embed_dim':vssm.EMBED_DIM,'depths':vssm.DEPTHS,
            'downsample':vssm.DOWNSAMPLE,'patchembed':vssm.PATCHEMBED,'patch_norm':vssm.PATCH_NORM}
        model = HierarchicalSCDInstance(pretrained=None, num_queries_per_scale=args.num_queries,
            clip_weights_path=args.clip_weights_path, **cfg_d).to(device)
        print(f"Loading: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Epoch: {ckpt.get('epoch','?')}")
        preds = run_inference(model, loader, device, args.score_thresh)
        icd = evaluator.evaluate_instance(preds, all_gt)
        print_icd(icd, "Instance Model — ICD Metrics")
        tgt, st = evaluate_per_class(preds, all_gt)
        print_per_class(tgt, st)

    elif args.mode == "pixel":
        if not args.pred_dir:
            print("ERROR: --pred_dir required"); sys.exit(1)
        icd = evaluator.evaluate_pixel(args.pred_dir, all_gt)
        print_icd(icd, "Pixel Model — ICD Metrics")

    elif args.mode == "scd":
        if not args.pred_dir:
            print("ERROR: --pred_dir required for scd mode"); sys.exit(1)
        icd = evaluator.evaluate_scd(args.pred_dir, all_gt,
            pred_format=args.pred_format, pre_dir=args.pre_dir, post_dir=args.post_dir)
        print_icd(icd, "SCD Model — ICD Metrics")

    if args.output_json:
        save = {str(k):v for k,v in icd.items() if isinstance(v,dict)}
        save['mAP'] = icd.get('mAP',0); save['mAP_50'] = icd.get('mAP_50',0); save['mAP_75'] = icd.get('mAP_75',0)
        with open(args.output_json,'w') as f: json.dump(save, f, indent=2)
        print(f"\nSaved: {args.output_json}")
