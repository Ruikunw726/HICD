# HICD v7 - Hierarchical Instance Change Detection

Dual-branch change detection with interaction layer.

## Architecture

```
T1, T2 -> LWGANet (shared) -> NFA -> TFM (fused + T1/T2 independent)
  -> Task Adapters
  -> Instance Branch (DETR) + Pixel Branch (FPN)
  -> Interaction Layer
  -> Output: boxes + target class + change type + damage level
```

### New in V7

- **Temporal Fusion**: preserves T1/T2 independent features for interaction
- **ChangeGuidedAttention**: change heatmap guides instance branch attention (gate=0 initially)
- **BoxConsistencyLoss**: detection boxes constrain pixel branch consistency
- **ChangeTypeClassifier**: T1/T2 feature comparison -> change type + damage level
- **Full-image training**: no 256 patch splitting, resize to fixed training size
- **Proper CSV logging**: all metrics saved to training_log.csv

### Active Heads (dataset-specific)

```yaml
# xbd.yaml
active_heads:
  instance_detection: true
  change_type: false          # xBD has no change type annotation
  damage_level: true          # xBD has 5-level damage annotation
  pixel_target: false
  pixel_state: false
```

## Training

```bash
cd HICD_v7
export PYTHONPATH=/path/to/home:$PYTHONPATH

python script/train_full_v7.py \
  --dataset xbd \
  --data_dir /path/to/xbd \
  --backbone L1 \
  --pretrained_weight_path /path/to/lwganet_l1_e299.pth \
  --batch_size 2 \
  --max_epochs 200 \
  --learning_rate 1e-4 \
  --use_amp \
  --num_workers 4 \
  --img_size 1024 \
  --exp_name v7_xbd
```

## Outputs

- `pred_boxes`: (B, num_queries, 4) - detection boxes
- `pred_logits`: (B, num_queries, N_targets) - target class
- `pred_state_logits`: (B, num_queries, N_states) - damage state
- `pred_change_type`: (B, num_queries, 5) - change type
- `pred_damage_level`: (B, num_queries, 4) - damage level
- `change_heatmap`: (B, 1, H/4, W/4) - change heatmap
- `target_type_map`: (B, N_states, H/4, W/4) - target type map