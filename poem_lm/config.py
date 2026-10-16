# -*- coding: utf-8 -*-
"""超参数配置：模型结构与训练参数集中管理，附两套预设。"""

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    d_model: int = 256      # 词向量/注意力维度
    n_heads: int = 8        # 注意力头数(须整除 d_model)
    n_layers: int = 6       # 解码器层数
    d_ff: int = 1024        # 前馈网络隐藏层维度
    block_size: int = 32    # 最大序列长度
    dropout: float = 0.1
    tie_embeddings: bool = False  # 输出层与词嵌入共享权重(省参数+正则化)


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 8e-4        # 峰值学习率(余弦退火后降到 min_lr)
    min_lr: float = 8e-5
    warmup_steps: int = 500  # 线性预热步数
    epochs: int = 12
    weight_decay: float = 0.1   # 仅作用于权重矩阵(见 Trainer 参数分组)
    grad_clip: float = 1.0
    amp: bool = True         # 混合精度训练(GPU 上显著加速)
    val_batches: int = 20   # 每轮验证最多前向多少个 batch
    log_interval: int = 50  # 每多少步打印一次训练日志
    patience: int = 3       # 早停：val_loss 连续多少轮未改善就停止训练
    use_compile: bool = True  # 是否启用 torch.compile(不可用时自动回退 eager)
    seed: int = 42


# 预设：small 快速验证流程；base 常规训练；large 突破 base 的 val_loss 平台期
PRESETS = {
    "small": (
        ModelConfig(d_model=128, n_heads=4, n_layers=3, d_ff=512, block_size=32, dropout=0.1),
        TrainConfig(batch_size=64, lr=1e-3, epochs=4, warmup_steps=200),
    ),
    "base": (
        ModelConfig(d_model=256, n_heads=8, n_layers=6, d_ff=1024, block_size=32, dropout=0.1),
        # batch 64->256 后按平方根法则把 lr 8e-4 提到 1.5e-3；每轮步数约 1790 -> 447，
        # warmup 300 约占全程 7%；epochs 12->8 配合早停(patience=3)，通常更早收敛且不过拟合
        TrainConfig(batch_size=256, lr=1.5e-3, epochs=8, warmup_steps=300,
                    val_batches=20, patience=3, log_interval=50),
    ),
    # 突破 base(val_loss~3.59 平台期)的三项关键变化：
    #   1) 容量 7.4M -> 约 16M(d_model 384 / 8 层 / d_ff 1536)；
    #   2) block_size 32 -> 64：一首七言绝句约 40 token，32 会截断跨句上下文，
    #      64 能让模型看到整首诗，学对仗、韵脚与句间衔接；
    #   3) 输出层与词嵌入共享权重(tie)，等效正则化，通常直接降低 val_loss。
    # block 翻倍后 batch 256->192 控显存；lr 相应微降。须从头训练(结构不兼容旧 ckpt)。
    # epochs=40：轮数写在这里(不在命令行传)，patience=5 保证无效时自动早停。
    "large": (
        ModelConfig(d_model=384, n_heads=8, n_layers=8, d_ff=1536, block_size=64,
                    dropout=0.15, tie_embeddings=True),
        TrainConfig(batch_size=192, lr=1.2e-3, epochs=80, warmup_steps=300,
                    min_lr=8e-5, val_batches=100, patience=5, log_interval=50),
    ),
}


def get_preset(name: str):
    if name not in PRESETS:
        raise ValueError(f"未知预设 {name!r}，可选: {list(PRESETS)}")
    m, t = PRESETS[name]
    return asdict(m), asdict(t)


# 模型结构字段：用来判断一个 checkpoint 是用哪个预设训出来的
_MODEL_CFG_KEYS = ("d_model", "n_heads", "n_layers", "d_ff", "block_size",
                   "dropout", "tie_embeddings")


def preset_for_model_cfg(model_cfg):
    """按模型结构反查预设名(认不出返回 None)。

    续训时 checkpoint 自带 model_cfg，用它反查预设，就不必手写 --preset
    去对齐结构(写错会直接 shape mismatch 报错)。
    """
    if not model_cfg:
        return None
    for name, (mcfg, _) in PRESETS.items():
        if all(model_cfg.get(k) == getattr(mcfg, k) for k in _MODEL_CFG_KEYS):
            return name
    return None
