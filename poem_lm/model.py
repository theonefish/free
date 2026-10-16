# -*- coding: utf-8 -*-
"""模型模块：多头自注意力、位置编码、解码器层与完整模型。

结构要点：
  * Decoder-Only(仅解码器)：作诗是纯生成任务，因果掩码保证每个位置
    只能看到之前的字，直接建模 p(下一字 | 之前所有字)；
  * Pre-Norm 残差结构：x = x + SubLayer(LayerNorm(x))，比 Post-Norm
    在层数较深时训练更稳定；
  * 注意力权重默认不保存(省显存/时间)，需要可视化时传 return_attn=True。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD_IDX


class MultiHeadAttention(nn.Module):
    """
    多头缩放点积注意力：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    把 d_model 拆成 n_heads 份独立计算再拼接，让模型同时关注不同位置、
    不同子空间的关系。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # (b, T, d_model) -> (b, heads, T, d_k)
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x, mask=None, return_attn=False):
        """
        x:    (batch, seq_len, d_model)
        mask: (batch, 1, seq_len, seq_len)，True 表示允许注意力
        return_attn: 仅注意力可视化需要，走手写路径(性能较低)
        """
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))
        if return_attn or mask is not None:
            # 手写路径：需要 attn 权重(可视化)或自定义掩码(如 padding)时使用
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            attn = self.dropout(F.softmax(scores, dim=-1))
            out = torch.matmul(attn, v)                       # (b, heads, T, d_k)
            attn = attn.detach()
        else:
            # 快路径：纯因果掩码交给 SDPA，可走 flash / mem-efficient kernel，
            # 省去手写实现的多次 kernel 启动
            attn = None
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(x.shape)  # 拼接多头
        out = self.w_o(out)
        return (out, attn) if return_attn else out


class PositionalEncoding(nn.Module):
    """
    正弦位置编码：注意力本身不区分位置，需要显式注入"第几个字"。
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    固定公式(不可学习)是原论文做法，可外推更长序列。
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class DecoderBlock(nn.Module):
    """
    一个解码器层 = 多头自注意力 + 前馈网络，各自包"残差连接 + LayerNorm"(Pre-Norm)：
        x = x + Attention(LayerNorm(x))      # 收集上下文信息
        x = x + FFN(LayerNorm(x))            # 逐位置非线性变换
    【残差连接】让梯度绕过深层直接回传；【LayerNorm】稳定每层输入分布。
    若要扩展成编码器-解码器(对句模型)：在注意力后加一个 CrossAttention
    (query 来自解码器，key/value 来自编码器输出)即可。
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop1(self.attn(self.norm1(x), mask))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class PoemTransformer(nn.Module):
    """Decoder-Only 写诗模型：嵌入 -> 位置编码 -> N 个解码器层 -> 输出词表分布。"""

    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6,
                 d_ff=1024, block_size=32, dropout=0.1, tie_embeddings=False):
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(d_model, block_size, dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)          # Pre-Norm 结构末尾的归一化
        self.head = nn.Linear(d_model, vocab_size)
        if tie_embeddings:
            # 输入嵌入与输出投影共享同一矩阵(GPT-2 做法)：省 vocab*d_model 参数，
            # 且嵌入被迫同时"解释"输入与输出空间，对小语料是有效的正则化
            self.head.weight = self.token_emb.weight
        # 预计算因果掩码(下三角)，避免每个 forward 重建 T×T 矩阵；
        # persistent=False -> 不进 state_dict，不影响旧 checkpoint 加载
        causal = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal, persistent=False)

    def make_mask(self, idx):
        """
        组合两种掩码：
            因果掩码(下三角)：位置 t 只能看到 <= t 的位置，防止"偷看未来"；
            填充掩码        ：<pad> 位置作为 key 时被屏蔽。
        返回 (batch, 1, T, T) 布尔张量，True 表示允许注意力。

        注意：热路径(见 forward)不调用本方法——本项目输入不含 <pad>，
        因果掩码直接走 SDPA 的 is_causal 快路径；仅在需要处理带
        padding 的输入时使用本方法获取完整组合掩码。
        """
        T = idx.size(1)
        pad_ok = (idx != PAD_IDX)
        mask = self.causal_mask[:T, :T].unsqueeze(0) & pad_ok.unsqueeze(1)
        return mask.unsqueeze(1)

    def forward(self, idx, targets=None):
        """
        idx:     (batch, seq_len) 输入 token id
        targets: (batch, seq_len) 目标 token id(与输入错开一位)，提供则返回 loss
        """
        x = self.token_emb(idx) * math.sqrt(self.d_model)   # 嵌入(按论文缩放)
        x = self.pos_enc(x)
        # 本项目的 token 流数据集无 <pad>(batch 为连续切片，prompt 也只有 BOS+汉字)，
        # 无需逐位构造组合掩码：因果掩码交给 SDPA 的 is_causal 快路径。
        # 若未来输入可能含 padding，改回: mask = self.make_mask(idx) 并传给 block
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=PAD_IDX,
            )
        return logits, loss
