# -*- coding: utf-8 -*-
"""
Class Mapping — 所有类别定义的唯一数据源

设计依据: D:\CD\0617final\classes.csv
  - 16 种目标类型 (target_type), 6 种变化状态 (change_state)
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
    "Bridge",           # 4  桥梁
    "Highway",          # 5  主要公路
    "Building",         # 6  建筑物
    "Shelter",          # 7  飞机掩体
    "Tower",            # 8  塔台
    "Pier",             # 9  栈桥
    "Dock",             # 10 船坞
    "Tank",             # 11 大型罐体
    "Aircraft",         # 12 飞机
    "Vessel",           # 13 舰船
    "Crater",           # 14 弹坑
    "VehicleRevet",     # 15 车辆掩体
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
    4:  [0, 1, 2, 3, 4],      # Bridge:      NoChange, Damaged, Reduced, Added, Extended
    5:  [0, 1, 2, 3, 4],      # Highway:     NoChange, Damaged, Reduced, Added, Extended
    6:  [0, 1, 2, 3, 4],      # Building:    NoChange, Damaged, Reduced, Added, Extended
    7:  [0, 1, 2, 3, 4],      # Shelter:     NoChange, Damaged, Reduced, Added, Extended
    8:  [0, 1, 2, 3, 4],      # Tower:       NoChange, Damaged, Reduced, Added, Extended
    9:  [0, 1, 2, 3, 4],      # Pier:        NoChange, Damaged, Reduced, Added, Extended
    10: [0, 1, 2, 3, 4],      # Dock:        NoChange, Damaged, Reduced, Added, Extended
    11: [0, 1, 2, 3],         # Tank:        NoChange, Damaged, Reduced, Added (无 Extended)
    12: [0, 1, 2, 3, 5],      # Aircraft:    NoChange, Damaged, Reduced, Added, Replaced (无 Extended)
    13: [0, 1, 2, 3, 5],      # Vessel:      NoChange, Damaged, Reduced, Added, Replaced (无 Extended)
    14: [0],                   # Crater:      无状态 (Stateless)
    15: [0],                   # VehicleRevet: 无状态 (Stateless)
}

# ── CLIP 文本提示词 ──────────────────────────────────────────────
# 每个目标类型的多描述文本，用于 CLIP 编码器
CLIP_TEXT_PROMPTS = [
    "farmland, agricultural field, crop land",
    "runway, airstrip, landing strip",
    "taxiway, aircraft taxi path",
    "apron, aircraft parking area",
    "bridge, overpass, viaduct",
    "highway, main road, expressway",
    "building, structure, house",
    "aircraft shelter, hardened shelter",
    "control tower, observation tower",
    "pier, wharf, jetty",
    "dock, shipyard, dry dock",
    "fuel tank, storage tank, oil tank",
    "aircraft, plane, airplane",
    "vessel, ship, boat",
    "crater, bomb crater, impact crater",
    "vehicle revetment, protective berm",
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
    # Bridge
    18: (4, 0), 19: (4, 1), 20: (4, 2), 21: (4, 3), 22: (4, 4),
    # Highway
    23: (5, 0), 24: (5, 1), 25: (5, 2), 26: (5, 3), 27: (5, 4),
    # Building
    28: (6, 0), 29: (6, 1), 30: (6, 2), 31: (6, 3), 32: (6, 4),
    # Shelter
    33: (7, 0), 34: (7, 1), 35: (7, 2), 36: (7, 3), 37: (7, 4),
    # Tower
    38: (8, 0), 39: (8, 1), 40: (8, 2), 41: (8, 3), 42: (8, 4),
    # Pier
    43: (9, 0), 44: (9, 1), 45: (9, 2), 46: (9, 3), 47: (9, 4),
    # Dock
    48: (10, 0), 49: (10, 1), 50: (10, 2), 51: (10, 3), 52: (10, 4),
    # Tank
    53: (11, 0), 54: (11, 1), 55: (11, 2), 56: (11, 3),
    # Aircraft
    57: (12, 0), 58: (12, 1), 59: (12, 2), 60: (12, 3), 61: (12, 5),
    # Vessel
    62: (13, 0), 63: (13, 1), 64: (13, 2), 65: (13, 3), 66: (13, 5),
    # Crater
    67: (14, 0),
    # Vehicle Revetment
    68: (15, 0),
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
