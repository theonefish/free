# -*- coding: utf-8 -*-
"""训练模块：Trainer 封装 AMP(bf16/fp16 自适应)、fused AdamW、torch.compile
(不可用时自动回退 eager)、预热+余弦学习率、验证集评估、早停、checkpoint
保存/恢复、训练日志与周期性作诗演示。"""

import math
import time
from pathlib import Path

import torch

from .generate import generate_line


def latest_checkpoint(ckpt_dir="checkpoints"):
    """返回续训用的 checkpoint 路径：优先 last.pt(每轮结束都写，带最新优化器
    状态与步数)，没有则退回 best.pt；都没有返回 None。—— 供 --continue 自动找断点。"""
    ckpt_dir = Path(ckpt_dir)
    for name in ("last.pt", "best.pt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    return None


class Trainer:
    def __init__(self, model, train_loader, val_loader, train_cfg, vocab,
                 device="cuda", ckpt_dir="checkpoints", model_cfg=None):
        if isinstance(device, str):
            device = torch.device(device)
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg
        self.vocab = vocab
        self.device = device
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        # 模型结构(随 checkpoint 一起保存)：作诗时可按它重建模型，
        # 不再依赖"当前用的是哪个预设"
        self.model_cfg = model_cfg

        self.use_amp = train_cfg.get("amp", True) and device.type == "cuda"
        # ---- 全局开关：Ampere+ GPU 的 fp32 matmul 允许 TF32(几乎无损，显著提速) ----
        torch.set_float32_matmul_precision("high")
        # ---- AMP 精度：bf16 动态范围大、无需 GradScaler；不支持的旧卡退回 fp16 ----
        if self.use_amp:
            self.amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                              else torch.float16)
        else:
            self.amp_dtype = torch.float32
        # 参数分组：权重矩阵(ndim>=2)加 weight_decay，偏置/LayerNorm 不加
        decay, no_decay = [], []
        for p in model.parameters():
            (decay if p.dim() >= 2 else no_decay).append(p)
        # fused=True：CUDA 上把 AdamW 的多个 kernel 融合为一个(需 fp32 参数，AMP 下满足)
        self.opt = torch.optim.AdamW([
            {"params": decay, "weight_decay": train_cfg.get("weight_decay", 0.1)},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=train_cfg["lr"], betas=(0.9, 0.95),
            fused=self.device.type == "cuda")
        # 仅 fp16 需要 GradScaler 做损失缩放；bf16 不需要
        self.scaler = (torch.amp.GradScaler("cuda", enabled=self.use_amp)
                       if (self.use_amp and self.amp_dtype == torch.float16) else None)
        # ---- torch.compile：Windows 官方 wheel 默认不带 Triton，inductor 可能
        # 不可用；suppress_errors=True 保证首次 forward 编译失败时自动回退 eager ----
        self._eager_model = self.model   # 保留未编译引用：作诗演示/调试用
        if train_cfg.get("use_compile", True):
            self.model = self._maybe_compile(self.model)
        self.step = 0
        self.best_val = float("inf")

    def _maybe_compile(self, model):
        """尽力启用 torch.compile；任何失败都安全回退 eager，不中断训练。"""
        if not hasattr(torch, "compile"):
            print("[compile] 当前 PyTorch 版本不支持 torch.compile，使用 eager")
            return model
        torch._dynamo.config.suppress_errors = True
        try:
            print("[compile] torch.compile 已启用(inductor；不可用时自动回退 eager)")
            return torch.compile(model)
        except Exception as e:
            print(f"[compile] 不可用，使用 eager: {e}")
            return model

    # -------------------------------------------------- 学习率：线性预热 + 余弦退火
    def lr_at(self, step, total_steps):
        cfg = self.cfg
        if step < cfg["warmup_steps"]:
            return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
        progress = (step - cfg["warmup_steps"]) / max(1, total_steps - cfg["warmup_steps"])
        coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return cfg["min_lr"] + coeff * (cfg["lr"] - cfg["min_lr"])

    # -------------------------------------------------- 训练 / 验证
    def train_one_epoch(self, total_steps):
        self.model.train()
        t0, tokens_seen = time.time(), 0
        for x, y in self.train_loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            for g in self.opt.param_groups:
                g["lr"] = self.lr_at(self.step, total_steps)

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                _, loss = self.model(x, y)

            self.opt.zero_grad(set_to_none=True)
            if self.scaler is not None:      # fp16：缩放反向 -> 反缩放 -> 裁剪
                self.scaler.scale(loss).backward()
                # 梯度裁剪(先反缩放再裁剪，保证阈值含义正确)
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.cfg["grad_clip"])
                self.scaler.step(self.opt)
                self.scaler.update()
            else:                            # bf16 / 无 AMP：直接反向 + 裁剪
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.cfg["grad_clip"])
                self.opt.step()

            self.step += 1
            tokens_seen += x.numel()
            if self.step % self.cfg["log_interval"] == 0:
                dt = time.time() - t0
                print(f"  step {self.step:6d} | loss {loss.item():.4f} | "
                      f"lr {self.opt.param_groups[0]['lr']:.2e} | "
                      f"{tokens_seen / dt:8.0f} tok/s", flush=True)
                t0, tokens_seen = time.time(), 0

    @torch.no_grad()
    def evaluate(self):
        """在验证集上计算平均 loss(最多前向 val_batches 个 batch)。"""
        self.model.eval()
        losses, n = 0.0, 0
        for i, (x, y) in enumerate(self.val_loader):
            if i >= self.cfg["val_batches"]:
                break
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                _, loss = self.model(x, y)
            losses += loss.item()
            n += 1
        self.model.train()
        return losses / max(1, n)

    # -------------------------------------------------- checkpoint
    def save_checkpoint(self, name, val_loss=None):
        # 用未编译的原始模型存取 state_dict：compile 包装器与原始模块共享
        # 同一份参数，这样可避免不同 PyTorch 版本下包装器的任何前缀差异
        ckpt = {
            "model": self._eager_model.state_dict(),
            "opt": self.opt.state_dict(),
            "step": self.step,
            "best_val": self.best_val,
            "vocab": self.vocab.itos,
            "model_cfg": self.model_cfg,
        }
        torch.save(ckpt, self.ckpt_dir / name)
        if val_loss is not None:
            print(f"  [checkpoint] {name} (val_loss={val_loss:.4f})")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self._eager_model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            self.opt.load_state_dict(ckpt["opt"])
        self.step = ckpt.get("step", 0)
        self.best_val = ckpt.get("best_val", float("inf"))
        print(f"  [checkpoint] 已恢复 {path} (step={self.step}, best_val={self.best_val:.4f})")

    # -------------------------------------------------- 主循环
    def fit(self, resume=None):
        if resume:
            self.load_checkpoint(resume)
        steps_per_epoch = len(self.train_loader)
        # 续训时让余弦调度覆盖"剩余训练量"：否则恢复点已越过调度终点，
        # 后续轮次全部以 min_lr 运行，等于浪费(上次 16 轮续训的后 8 轮即如此)
        total_steps = self.step + steps_per_epoch * self.cfg["epochs"]
        amp_info = (f"开({str(self.amp_dtype).replace('torch.', '')})"
                    if self.use_amp else "关")
        compiled = self.model is not self._eager_model
        print(f"[训练] 设备 {self.device} | AMP {amp_info} | "
              f"compile {'开' if compiled else '关'} | "
              f"每轮 {steps_per_epoch} 步 | 共 {total_steps} 步", flush=True)

        bad_epochs = 0
        patience = self.cfg.get("patience", 0)
        # 早停只看"本轮训练内部"的改善(run_best 每次 fit 重置为 inf)：
        # 续训时若提高/调整学习率，val_loss 会先升后降，用历史 best_val
        # 做耐心计数会造成"假早停"(前几轮必然连续不达历史最佳)
        run_best = float("inf")
        for epoch in range(1, self.cfg["epochs"] + 1):
            t0 = time.time()
            self.train_one_epoch(total_steps)
            val_loss = self.evaluate()
            mark = ""
            # best.pt 的保存仍以全局历史最佳为准，永远不会被更差的权重覆盖
            if val_loss < self.best_val - 1e-4:
                self.best_val = val_loss
                self.save_checkpoint("best.pt", val_loss)
                mark = "  *best"
            if val_loss < run_best - 1e-4:
                run_best = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
            # 用未编译模型作诗：避免变长序列触发 Dynamo 反复重编译
            demo = generate_line(self._eager_model, self.vocab, meter="七言",
                                 temperature=1.0, top_k=8)
            print(f"Epoch {epoch:3d}/{self.cfg['epochs']} | val_loss {val_loss:.4f} | "
                  f"{time.time() - t0:5.0f}s | 试作: {demo}{mark}", flush=True)
            if patience and bad_epochs >= patience:
                print(f"[早停] 本轮 val_loss 连续 {bad_epochs} 轮未改善"
                      f"(run_best={run_best:.4f})，停止训练")
                break
        self.save_checkpoint("last.pt")
        print(f"[训练] 完成，最佳 val_loss = {self.best_val:.4f}")
