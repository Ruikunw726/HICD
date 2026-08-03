# -*- coding: utf-8 -*-
"""
Class Mapping — 数据集驱动的类别定义

设计: 每个数据集一个 YAML 配置文件, 模型按 dataset name 自动加载。
不同数据集 (0617final / xbd / ...) 自动适配, 不写死任何类别。

用法:
    # 方式1: 按名字加载 (推荐)
    from class_mapping import DatasetConfig
    cfg = DatasetConfig.load("0617final")   # 自动找 configs/datasets/0617final.yaml

    # 方式2: 加载默认 (向后兼容, 等价于 load("0617final"))
    from class_mapping import NUM_TARGETS, NUM_STATES, ...
"""

import torch
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

# 配置文件目录 (相对于本文件)
_CONFIG_DIR = Path(__file__).parent.parent / "configs" / "datasets"


@dataclass
class DatasetConfig:
    """数据集类别配置."""

    dataset: str = ""
    num_targets: int = 0
    num_states: int = 0
    target_names: List[str] = field(default_factory=list)
    state_names: List[str] = field(default_factory=list)
    clip_text_prompts: List[str] = field(default_factory=list)
    state_clip_prompts: List[str] = field(default_factory=list)
    target_valid_states: Dict[int, List[int]] = field(default_factory=dict)
    train_id_map: Dict[int, List[int]] = field(default_factory=dict)

    @classmethod
    def load(cls, name: str) -> "DatasetConfig":
        """按数据集名字加载配置.

        Args:
            name: 数据集名, 如 "0617final", "xbd"
                  会自动查找 configs/datasets/{name}.yaml
        """
        path = _CONFIG_DIR / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Dataset config not found: {path}")
        return cls.from_yaml(path)

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetConfig":
        """从 YAML 文件加载配置."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        cfg = cls()
        cfg.dataset = data.get('dataset', '')
        cfg.num_targets = data.get('num_targets', 0)
        cfg.num_states = data.get('num_states', 0)
        cfg.target_names = data.get('target_names', [])
        cfg.state_names = data.get('state_names', [])
        cfg.clip_text_prompts = data.get('clip_text_prompts', [])
        cfg.state_clip_prompts = data.get('state_clip_prompts', [])

        # target_valid_states: yaml keys 是 str, 转成 int
        raw_vs = data.get('target_valid_states', {})
        cfg.target_valid_states = {int(k): v for k, v in raw_vs.items()}

        # train_id_map: 值是 [target_idx, state_idx]
        raw_map = data.get('train_id_map', {})
        cfg.train_id_map = {int(k): tuple(v) for k, v in raw_map.items()}

        return cfg

    def get_valid_state_mask(self) -> torch.Tensor:
        """返回 (num_targets, num_states) 的有效性矩阵."""
        mask = torch.zeros(self.num_targets, self.num_states)
        for t, states in self.target_valid_states.items():
            for s in states:
                if t < self.num_targets and s < self.num_states:
                    mask[t, s] = 1.0
        return mask

    def train_id_to_target_state(self, train_id: int):
        """train_id → (target_idx, state_idx)"""
        return self.train_id_map.get(train_id, None)

    def print_summary(self):
        """打印类别摘要."""
        print(f"[{self.dataset}] {self.num_targets} targets × {self.num_states} states")
        for i, name in enumerate(self.target_names):
            states = [self.state_names[s] for s in self.target_valid_states.get(i, [])
                      if s < len(self.state_names)]
            print(f"  [{i:2d}] {name:15s} → {states}")

        total = sum(len(v) for v in self.target_valid_states.values())
        print(f"  Valid (target, state) combos: {total}")
        print(f"  Flat classes (incl. background): {total + 1}")


# ═══════════════════════════════════════════════════════════════════
# 向后兼容: 模块级常量 (默认加载 0617final)
# 旧代码可以继续 import 这些常量
# ═══════════════════════════════════════════════════════════════════

def _load_default():
    """加载默认数据集配置 (0617final)."""
    try:
        return DatasetConfig.load("0617final")
    except FileNotFoundError:
        return DatasetConfig()

_default_cfg = _load_default()

# 模块级常量 (向后兼容)
TARGET_NAMES = _default_cfg.target_names
STATE_NAMES = _default_cfg.state_names
NUM_TARGETS = _default_cfg.num_targets
NUM_STATES = _default_cfg.num_states
TARGET_VALID_STATES = _default_cfg.target_valid_states
CLIP_TEXT_PROMPTS = _default_cfg.clip_text_prompts
_TRAIN_ID_MAP = _default_cfg.train_id_map


def train_id_to_target_state(train_id):
    """train_id → (target_idx, state_idx)"""
    return _TRAIN_ID_MAP.get(train_id, None)


def get_valid_state_mask():
    """返回 (NUM_TARGETS, NUM_STATES) 的有效性矩阵."""
    return _default_cfg.get_valid_state_mask()


def print_class_summary():
    """打印类别摘要."""
    _default_cfg.print_summary()


if __name__ == "__main__":
    import sys
    datasets_dir = Path(__file__).parent / "configs" / "datasets"
    for yaml_file in sorted(datasets_dir.glob("*.yaml")):
        cfg = DatasetConfig.from_yaml(yaml_file)
        cfg.print_summary()
        print()


