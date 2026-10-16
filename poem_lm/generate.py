# -*- coding: utf-8 -*-
"""生成模块：自回归采样(温度 / top-k / top-p / 重复惩罚)、体裁控制与等长强制解码。"""

import torch
import torch.nn.functional as F

from .data import BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX

# 生成过程中禁止出现的特殊 token(保证可见输出全是汉字)
_SPECIAL_IDS = (PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX)

# 默认重复惩罚：>1 时惩罚上下文中已出现的字，缓解"不知/不是"式复读
DEFAULT_REP_PENALTY = 1.5


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None,
             top_p=None, repetition_penalty=1.0, exact_len=None):
    """
    自回归生成：每步用当前全部 token 预测下一个字的分布，采样后接续，重复
    直到写出 <eos> 或达到长度上限。

    temperature      : >1 更随机，<1 更保守
    top_k            : 每步只在概率最高的 k 个字中采样
    top_p            : 核采样，只从累积概率达 p 的最小候选集中采样
    repetition_penalty: >1 时惩罚上下文中已出现的字，缓解"复读"
    exact_len        : 若指定(如 5 或 7)，则严格生成 exact_len 个汉字后收句——
                       写满前禁止 <eos> 及一切特殊 token，写满后强制 <eos>。
                       这保证了五言/七言诗句长短统一。
    """
    model.eval()
    n_chars = 0
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]          # 上下文超长则截断
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        if repetition_penalty != 1.0:                   # 重复惩罚
            for token in set(idx_cond[0].tolist()):
                logits[:, token] /= repetition_penalty

        if exact_len is not None:                       # 等长强制约束
            if n_chars < exact_len:
                # 没写够：禁止所有特殊 token(含 <eos>)，只允许出汉字
                logits[:, list(_SPECIAL_IDS)] = float("-inf")
            else:
                # 写够了：强制收句
                logits[:, :] = float("-inf")
                logits[:, EOS_IDX] = 0.0

        if top_k:                                       # top-k 过滤
            k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, k).values[:, -1:]
            logits = torch.where(logits < threshold,
                                 torch.full_like(logits, float("-inf")), logits)
        if top_p:                                       # top-p(核)过滤
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone()  # 保留第一个超阈值 token
            remove[..., 0] = False
            logits.scatter_(-1, sorted_idx,
                            torch.where(remove, torch.full_like(sorted_logits, float("-inf")),
                                        sorted_logits))

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)   # 按概率采样
        idx = torch.cat([idx, next_id], dim=1)
        if next_id.item() == EOS_IDX:                       # 本实现 batch=1
            break
        if exact_len is None or next_id.item() not in _SPECIAL_IDS:
            n_chars += 1
    return idx


def generate_line(model, vocab, prompt="", meter=None, exact_len=None,
                  temperature=1.0, top_k=None, top_p=None,
                  repetition_penalty=DEFAULT_REP_PENALTY):
    """
    生成一句诗。

    meter   : '五言' / '七言' / None —— 句首插入对应体裁标记，引导模型
              按该句式写作(需语料带体裁标记，见 prepare_data.py)；
              None 且给了 5/7 字 prompt 时按 prompt 长度自动推断
    exact_len: 5 / 7 / None —— 强制本句恰好生成这么多个字(与 meter 配合
              保证长短统一)；None 时不限制，由模型自然决定
    prompt  : 非空时为条件生成(例如给上句对下句)。上下文按训练语料的
              格式拼成 [<bos> 体裁 上句 <eos> <bos> 体裁]，让模型在
              "见过"的分布里续写下句——直接把上句接在体裁标记后属于
              分布外输入，生成质量会明显变差
    """
    if meter is None and prompt:
        meter = {5: "五言", 7: "七言"}.get(len(prompt))
    ids = [BOS_IDX]
    if meter:
        ids.append(vocab.meter_id(meter))
    if prompt:
        # 上句完整放入(与语料逐句拼接的格式一致)，再开新句让模型续写
        ids.extend(vocab.encode(prompt))
        ids.append(EOS_IDX)
        ids.append(BOS_IDX)
        if meter:
            ids.append(vocab.meter_id(meter))
    idx = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    if exact_len is None and meter in ("五言", "七言"):
        exact_len = 5 if meter == "五言" else 7
    out = generate(model, idx, model.block_size - len(ids),
                   temperature=temperature, top_k=top_k, top_p=top_p,
                   repetition_penalty=repetition_penalty, exact_len=exact_len)
    # 只返回新生成的下句(跳过作为条件的上句部分)
    return vocab.decode(out[0, len(ids):].tolist())


def generate_poem(model, vocab, n_lines=4, meter=None, temperature=1.0,
                  top_k=None, top_p=None,
                  repetition_penalty=DEFAULT_REP_PENALTY):
    """
    生成一首 n_lines 句的小诗(绝句 4 句：奇数句逗号，偶数句句号)。

    meter: '五言'/'七言'/None。指定后全诗统一体裁、每句等长；
           None 时每句长度由模型自定(不推荐，会出现长短不一)。

    实现方式：整首诗在同一个增长的上下文里逐句生成(每句前补 <bos>+体裁标记，
    与训练格式一致)。这样重复惩罚和模型注意力都覆盖全诗，可减少"今日…今日"
    式的跨句重复，句子间也更有衔接感。
    """
    exact_len = 5 if meter == "五言" else 7 if meter == "七言" else None
    device = next(model.parameters()).device
    ids = [BOS_IDX]
    if meter:
        ids.append(vocab.meter_id(meter))
    lines = []
    for i in range(n_lines):
        if i > 0:                       # 第二句起：先补 <bos>(+体裁标记)再续写
            ids.append(BOS_IDX)
            if meter:
                ids.append(vocab.meter_id(meter))
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        before = len(ids)
        out = generate(model, idx, model.block_size, exact_len=exact_len,
                       temperature=temperature, top_k=top_k, top_p=top_p,
                       repetition_penalty=repetition_penalty)
        new_ids = out[0, before:].tolist()     # 本句新生成的 token
        lines.append(vocab.decode(new_ids))    # decode 遇 <eos> 停止
        ids.extend(new_ids)                    # 累积进上下文，供下一句续写
    return "\n".join(l + ("，" if i % 2 == 0 else "。") for i, l in enumerate(lines))
