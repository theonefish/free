# -*- coding: utf-8 -*-
"""数据模块：词表、token 流数据集、DataLoader 构建。

设计要点(相比单文件版本的结构优化)：
  * 语料预先构建成"扁平 token 流"([<bos>句<eos>] 首尾相接)并由 prepare_data.py
    保存为 .pt，训练时 O(1) 加载，无需重复解析 5.7 万首诗；
  * 训练样本 = 从 token 流中随机切取的固定长度窗口，batch 内无 padding，
    GPU 利用率远高于逐句补齐的方案；
  * 词表与语料统计通过文件传递，训练/生成脚本共享同一份 CharVocab。
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
# 体裁标记 token(在词表 itos 中位于 4 个基础特殊 token 之后)
METER_TOKENS = {"五言": "<五言>", "七言": "<七言>"}


class CharVocab:
    """字级词表：每个汉字一个 id，另加特殊 token(含体裁标记)。"""

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def encode(self, s: str):
        return [self.stoi.get(c, UNK_IDX) for c in s]

    def meter_id(self, name: str) -> int:
        """取体裁标记('五言'/'七言')的 token id。"""
        tok = METER_TOKENS.get(name)
        if tok is None or tok not in self.stoi:
            raise KeyError(f"词表中没有体裁标记 {name!r}，请重新运行 prepare_data.py")
        return self.stoi[tok]

    def decode(self, ids):
        """id 序列 -> 字符串：遇 <eos> 停止，跳过所有 <...> 特殊 token。"""
        chars = []
        for i in ids:
            i = int(i)
            if i == EOS_IDX:
                break
            tok = self.itos[i] if 0 <= i < len(self.itos) else ""
            if tok.startswith("<") and tok.endswith(">"):
                continue
            chars.append(tok)
        return "".join(chars)


class TokenStreamDataset(Dataset):
    """
    扁平 token 流上的分块窗口数据集。

    stream 形如: [<bos> 床前明月光 <eos> <bos> 疑是地上霜 <eos> ...]
    把流按 block_size 顺序切成不重叠的块，每块再错开一位得到 (输入, 目标)：
        输入 = chunk[:-1]，目标 = chunk[1:]
    即"下一字预测"。每个 epoch 恰好覆盖全语料一次，配合 DataLoader 的
    shuffle=True，每轮以不同顺序见到这些块。
    """

    def __init__(self, stream: torch.Tensor, block_size: int):
        assert stream.dtype == torch.int32
        self.stream = stream
        self.block_size = block_size

    def __len__(self):
        # 每块需要 block_size+1 个 token(输入+目标)
        return (self.stream.numel() - 1) // self.block_size

    def __getitem__(self, i):
        chunk = self.stream[i * self.block_size: (i + 1) * self.block_size + 1]
        return chunk[:-1].long(), chunk[1:].long()


def build_dataloaders(data_dir, batch_size, block_size, num_workers=0):
    """
    从 data_dir 读取 corpus_train.pt / corpus_val.pt / vocab.json，
    返回 (train_loader, val_loader, vocab)。

    num_workers 默认 0：本数据集的 __getitem__ 只是张量切片(微秒级)，
    worker 进程启动与进程间通信反而是净开销，GPU 训练时 0 最快。
    """
    data_dir = Path(data_dir)
    train_stream = torch.load(data_dir / "corpus_train.pt")
    val_stream = torch.load(data_dir / "corpus_val.pt")
    vocab = CharVocab.load(data_dir / "vocab.json")

    train_ds = TokenStreamDataset(train_stream, block_size)
    val_ds = TokenStreamDataset(val_stream, block_size)
    # shuffle=True -> 每轮以不同顺序见到各块，避免相邻 batch 相关
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    return train_loader, val_loader, vocab
