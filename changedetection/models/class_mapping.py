# -*- coding: utf-8 -*-
"""
Class Mapping — 所有类别定义的唯一数据源

设计依据: D:\CD\0617final\classes.csv
  - 10 种目标类型 (target_type), 6 种变化状态 (change_state)
  - 层级结构: 目标类型 → 变化状态（不是所有目标都有全部状态）
  - 背景 (train_id=0) 不参与检测

使用场景:
  - 模型检测头输出维度定义
  - 损失函数中的类别映射
  - 像素→实例转换脚本的类别索引
  - 推理结果的类别名称还原
  - CLIP 文本提示词
"""

# ── 目标类型 ─────────────────────────────────────────────────────
# 顺序与 classes.csv 中 target_en 的出现顺序一致
TARGET_NAMES = [
    "Farmland",         # 0  农田
    "Runway",           # 1  跑道
    "Taxiway",          # 2  滑行道
    "Apron",            # 3  停机坪
    "Highway",          # 4  主要公路
    "Building",         # 5  建筑物
    "Tank",             # 6  大型罐体
    "Aircraft",         # 7  飞机
    "Vessel",           # 8  舰船
    "Crater",           # 9  弹坑
]
NUM_TARGETS = len(TARGET_NAMES)  # 16

# ── 变化状态 ─────────────────────────────────────────────────────
STATE_NAMES = [
    "NoChange",    # 0  无变化
    "Damaged",     # 1  损毁
    "Reduced",     # 2  缩小
    "Added",       # 3  新增
    "Extended",    # 4  扩建
    "Replaced",    # 5  替换
]
NUM_STATES = len(STATE_NAMES)  # 6

# ── 层级有效性矩阵 ──────────────────────────────────────────────
# 值为 1 表示该目标类型的该状态合法
# 来源: classes.csv 中每种目标实际出现的 state
TARGET_VALID_STATES = {
    0:  [0, 1],                # Farmland:    NoChange, Damaged
    1:  [0, 1, 2, 3, 4],      # Runway:      NoChange, Damaged, Reduced, Added, Extended
    2:  [0, 1, 2, 3, 4],      # Taxiway:     NoChange, Damaged, Reduced, Added, Extended
    3:  [0, 1, 2, 3, 4],      # Apron:       NoChange, Damaged, Reduced, Added, Extended
    4:  [0, 1, 2, 3, 4],      # Highway:     NoChange, Damaged, Reduced, Added, Extended
    5:  [0, 1, 2, 3, 4],      # Building:    NoChange, Damaged, Reduced, Added, Extended
    6:  [0, 1, 2, 3],         # Tank:        NoChange, Damaged, Reduced, Added
    7:  [0, 1, 2, 3, 5],      # Aircraft:    NoChange, Damaged, Reduced, Added, Replaced
    8:  [0, 1, 2, 3, 5],      # Vessel:      NoChange, Damaged, Reduced, Added, Replaced
    9:  [0],                   # Crater:      无状态 (Stateless)
}

# ── CLIP 文本提示词 ──────────────────────────────────────────────
# 每个目标类型的多描述文本，用于 CLIP 编码器
CLIP_TEXT_PROMPTS = [
    "farmland, agricultural field, crop land",
    "runway, airstrip, landing strip",
    "taxiway, aircraft taxi path",
    "apron, aircraft parking area",
    "highway, main road, expressway",
    "building, structure, house",
    "fuel tank, storage tank, oil tank",
    "aircraft, plane, airplane",
    "vessel, ship, boat",
    "crater, bomb crater, impact crater",
]

# ── train_id → (target_idx, state_idx) 映射 ─────────────────────
# 从 classes.csv 自动推导，与 pixel_to_instance 脚本共享
_TRAIN_ID_MAP = {
    # Farmland
    1: (0, 0), 2: (0, 1),
    # Runway
    3: (1, 0), 4: (1, 1), 5: (1, 2), 6: (1, 3), 7: (1, 4),
    # Taxiway
    8: (2, 0), 9: (2, 1), 10: (2, 2), 11: (2, 3), 12: (2, 4),
    # Apron
    13: (3, 0), 14: (3, 1), 15: (3, 2), 16: (3, 3), 17: (3, 4),
    # Highway (was 5)
    23: (4, 0), 24: (4, 1), 25: (4, 2), 26: (4, 3), 27: (4, 4),
    # Building (was 6)
    28: (5, 0), 29: (5, 1), 30: (5, 2), 31: (5, 3), 32: (5, 4),
    # Tank (was 11)
    53: (6, 0), 54: (6, 1), 55: (6, 2), 56: (6, 3),
    # Aircraft (was 12)
    57: (7, 0), 58: (7, 1), 59: (7, 2), 60: (7, 3), 61: (7, 5),
    # Vessel (was 13)
    62: (8, 0), 63: (8, 1), 64: (8, 2), 65: (8, 3), 66: (8, 5),
    # Crater (was 14)
    67: (9, 0),
}


def train_id_to_target_state(train_id):
    """train_id → (target_idx, state_idx)"""
    return _TRAIN_ID_MAP.get(train_id, None)


def get_valid_state_mask():
    """
    Returns:
        mask: (NUM_TARGETS, NUM_STATES) float tensor
              mask[t, s] = 1 表示目标类型 t 的状态 s 合法
    """
    import torch
    mask = torch.zeros(NUM_TARGETS, NUM_STATES)
    for t, states in TARGET_VALID_STATES.items():
        for s in states:
            mask[t, s] = 1.0
    return mask


def print_class_summary():
    """打印类别摘要"""
    print(f"目标类型: {NUM_TARGETS} 种")
    for i, name in enumerate(TARGET_NAMES):
        states = [STATE_NAMES[s] for s in TARGET_VALID_STATES[i]]
        print(f"  [{i:2d}] {name:15s} → {states}")

    total_combos = sum(len(v) for v in TARGET_VALID_STATES.values())
    print(f"\n有效 (目标, 状态) 组合: {total_combos}")
    print(f"扁平化类别数 (含背景): {total_combos + 1}")


if __name__ == "__main__":
    print_class_summary()
