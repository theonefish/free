# -*- coding: utf-8 -*-
"""poem_lm：基于全唐诗的 Decoder-Only Transformer 写诗模型。

模块划分：
    config   超参数与预设
    data     词表 / token 流数据集 / DataLoader
    model    注意力、位置编码、解码器层、完整模型
    train    Trainer(AMP、余弦学习率、验证、checkpoint)
    generate 自回归采样与作诗
"""

from .config import (ModelConfig, TrainConfig, PRESETS, get_preset,
                     preset_for_model_cfg)
from .data import CharVocab, TokenStreamDataset, build_dataloaders
from .model import PoemTransformer
from .train import Trainer, latest_checkpoint
from .generate import generate, generate_line, generate_poem

__all__ = [
    "ModelConfig", "TrainConfig", "PRESETS", "get_preset", "preset_for_model_cfg",
    "CharVocab", "TokenStreamDataset", "build_dataloaders",
    "PoemTransformer", "Trainer", "latest_checkpoint",
    "generate", "generate_line", "generate_poem",
]
