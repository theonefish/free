# -*- coding: utf-8 -*-
"""
用 Transformer 写五言/七言诗句 —— 训练入口
============================================
本项目已接入全唐诗(chinese-poetry 数据集，57,378 首、约 48 万句五言/七言)，
代码重构为模块化结构：

    prepare_data.py          数据准备：下载全唐诗 -> 解压 -> 繁简转换 -> 构建语料
    poem_lm/
        config.py            超参数与预设(small/base)
        data.py              词表、token 流数据集、DataLoader
        model.py             多头注意力、位置编码、解码器层、完整模型
        train.py             Trainer(AMP 混合精度、预热+余弦学习率、验证、checkpoint)
        generate.py          自回归采样与作诗
    transformer写短诗.py      本文件：命令行入口(训练 / 断点续训 / 作诗)

用法(训练轮次等超参写在 poem_lm/config.py 的 PRESETS 里，直接运行即按配置训练)：
    python prepare_data.py                      # 第一步：准备数据(只需一次)
    python transformer写短诗.py                  # 训练(默认 large 预设，40 轮)并演示作诗
    python transformer写短诗.py --continue       # 继续训练：自动找最新断点、自动识别预设，
                                               #   只补训到 config 里 epochs 的目标总轮数
    python transformer写短诗.py --preset base    # 换用 base 预设(轻量、快)
    python transformer写短诗.py --preset small   # 小模型快速验证流程
    python transformer写短诗.py --epochs 20      # 临时覆盖训练轮数(续训时=目标总轮数)
    python transformer写短诗.py --resume checkpoints/best.pt   # 从指定断点恢复(再训 epochs 轮)
    python transformer写短诗.py --gen-only       # 只作诗(加载 best.pt，不训练)
    python transformer写短诗.py --meter 七言      # 全部作七言(五言/七言/auto)
    python transformer写短诗.py --prompt 明月松间照  # 条件生成：给上句对下句
    python transformer写短诗.py --no-compile    # 关闭 torch.compile(调试/对比用)
"""

import argparse
import random
import sys
from pathlib import Path

import torch

# Windows 控制台默认 GBK，无法输出生僻 CJK 字符会直接崩溃，统一转为 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from poem_lm import (PoemTransformer, Trainer, build_dataloaders,  # noqa: E402
                     generate_line, generate_poem, get_preset,
                     latest_checkpoint, preset_for_model_cfg)
from poem_lm.config import PRESETS  # noqa: E402

DATA_DIR = ROOT / "data"


def main():
    ap = argparse.ArgumentParser(description="全唐诗 Transformer 写诗模型")
    ap.add_argument("--preset", default="large", choices=list(PRESETS),
                    help="模型规模预设(默认 large，轮次见 poem_lm/config.py)")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    ap.add_argument("--batch-size", type=int, default=None, help="覆盖批大小")
    ap.add_argument("--lr", type=float, default=None, help="覆盖峰值学习率")
    ap.add_argument("--resume", default=None,
                    help="从指定 checkpoint 恢复训练(再训 config 的 epochs 轮)")
    ap.add_argument("--continue", dest="cont", action="store_true",
                    help="继续训练：自动找最新 checkpoint 并识别其预设，"
                         "只补训到 config 里 epochs 的目标总轮数")
    ap.add_argument("--gen-only", action="store_true", help="跳过训练，直接作诗")
    ap.add_argument("--no-compile", action="store_true",
                    help="关闭 torch.compile(调试/对比用)")
    ap.add_argument("--gen-from", default=None,
                    help="作诗使用的 checkpoint(默认 checkpoints/best.pt)")
    ap.add_argument("--samples", type=int, default=3, help="作诗演示数量")
    ap.add_argument("--meter", default="auto", choices=["auto", "五言", "七言"],
                    help="作诗体裁：auto=每首随机选一种且全诗统一；五言/七言=全部统一")
    ap.add_argument("--rep-penalty", type=float, default=1.5,
                    help="重复惩罚：>1 抑制已出现的字复读，越大越多样(默认 1.5)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--prompt", default=None,
                    help="条件生成的上句(默认演示'明月松间照'等两例)")
    args = ap.parse_args()

    torch.manual_seed(42)
    random.seed(42)

    if not (DATA_DIR / "corpus_train.pt").exists():
        print("未找到训练语料，请先运行:  python prepare_data.py")
        sys.exit(1)

    if args.cont and args.resume:
        print("错误: --continue 与 --resume 只能二选一")
        sys.exit(1)

    # --continue：自动定位最新断点并识别其预设，用户无需手写 --resume/--preset
    ckpt_meta = None
    if args.cont and not args.gen_only:
        ckpt_path = latest_checkpoint(ROOT / "checkpoints")
        if ckpt_path is None:
            print("[续训] checkpoints/ 下没有可用断点，将从零开始训练")
        else:
            ckpt_meta = torch.load(ckpt_path, map_location="cpu")
            detected = preset_for_model_cfg(ckpt_meta.get("model_cfg"))
            if detected is None:
                print(f"[续训] 无法识别 {ckpt_path.name} 的模型结构(可能是旧版 "
                      f"checkpoint)，请改用 --resume 并手动指定 --preset")
                sys.exit(1)
            if detected != args.preset:
                print(f"[续训] 依据断点结构自动切换预设: {args.preset} -> {detected}")
            args.preset = detected
            args.resume = str(ckpt_path)
            print(f"[续训] 断点 {ckpt_path.name} | step {ckpt_meta.get('step', 0)} | "
                  f"历史最佳 val_loss {ckpt_meta.get('best_val', float('inf')):.4f}")

    model_cfg, train_cfg = get_preset(args.preset)
    if args.no_compile:
        train_cfg["use_compile"] = False
    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size
    if args.lr:
        train_cfg["lr"] = args.lr
    print(f"[配置] 预设 {args.preset} | 训练 {train_cfg['epochs']} 轮"
          f"(改轮次请编辑 poem_lm/config.py 的 PRESETS)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[环境] torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
              f"bf16 {'支持' if torch.cuda.is_bf16_supported() else '不支持(将用 fp16)'}")
    train_loader, val_loader, vocab = build_dataloaders(
        DATA_DIR, batch_size=train_cfg["batch_size"],
        block_size=model_cfg["block_size"])

    # --continue 把配置里的 epochs 当"目标总轮数"：只补训剩余轮数，余弦调度正好
    # 收在原目标终点上(不会把学习率重新拉高再训一遍)
    if ckpt_meta is not None:
        done = ckpt_meta.get("step", 0) // max(1, len(train_loader))
        target = train_cfg["epochs"]
        remaining = target - done
        if remaining <= 0:
            print(f"[续训] 断点已训约 {done} 轮，达到目标 {target} 轮；如需继续，"
                  f"请把 config.py 中 {args.preset} 的 epochs 调大，"
                  f"或用 --epochs 指定新的目标总轮数")
            sys.exit(0)
        train_cfg["epochs"] = remaining
        print(f"[续训] 已训约 {done} 轮 -> 目标 {target} 轮，本次补训 {remaining} 轮")

    model = PoemTransformer(len(vocab), **model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 预设 {args.preset} | 词表 {len(vocab)} | 参数量 {n_params / 1e6:.2f} M")

    if not args.gen_only:
        trainer = Trainer(model, train_loader, val_loader, train_cfg, vocab,
                          device=device, ckpt_dir=ROOT / "checkpoints",
                          model_cfg=model_cfg)
        trainer.fit(resume=args.resume)

    # 加载 checkpoint 用于作诗演示(--gen-only 时同样需要，否则是随机权重)
    ckpt_path = Path(args.gen_from) if args.gen_from else ROOT / "checkpoints" / "best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        saved_cfg = ckpt.get("model_cfg")
        if saved_cfg and saved_cfg != model_cfg:
            # checkpoint 自带结构信息：按它重建模型，避免预设与权重不匹配
            print(f"[模型] 按 checkpoint 记录的结构重建: {saved_cfg}")
            model = PoemTransformer(len(vocab), **saved_cfg)
        try:
            model.load_state_dict(ckpt["model"])
        except RuntimeError as e:
            detail = next((l.strip() for l in str(e).splitlines()
                           if "size mismatch" in l or "Missing key" in l),
                          str(e).splitlines()[0].strip())
            print(f"错误: {ckpt_path} 与预设 {args.preset} 结构不匹配，"
                  f"请用 --preset 指定训练该权重时所用的预设。")
            print(f"      {detail}")
            sys.exit(1)
        print(f"[模型] 已加载 {ckpt_path}")
    elif args.gen_only:
        print(f"错误: 找不到 checkpoint {ckpt_path}，请先训练")
        sys.exit(1)

    model.to(device).eval()
    print("\n================ 模型作品 ================")
    for i in range(args.samples):
        # auto: 每首诗随机选一种体裁，全诗统一(绝句体裁一致才像一首诗)
        meter = random.choice(["五言", "七言"]) if args.meter == "auto" else args.meter
        print(f"\n--- 第 {i + 1} 首({meter}) ---")
        print(generate_poem(model, vocab, n_lines=4, meter=meter,
                            temperature=args.temperature, top_k=args.top_k,
                            repetition_penalty=args.rep_penalty))
    print("\n--- 条件生成(给上句，试对下句) ---")
    # meter 按上句字数自动选择：五言对五言、七言对七言(体裁必须一致)
    prompts = [args.prompt] if args.prompt else ("明月松间照", "两岸猿声啼不住")
    for prompt in prompts:
        meter = "五言" if len(prompt) == 5 else "七言"
        reply = generate_line(model, vocab, prompt=prompt, meter=meter,
                              repetition_penalty=args.rep_penalty)
        print(f"上句[{prompt}]({meter}) -> 下句[{reply}]")


if __name__ == "__main__":
    main()
