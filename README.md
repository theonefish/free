# gen-poem：全唐诗 Transformer 写诗模型

基于 PyTorch 从零实现的字符级 Transformer 语言模型，在全唐诗（chinese-poetry 数据集，57,378 首、约 48 万句五言/七言）上训练，可生成五言/七言绝句，并支持"给上句对下句"的条件生成。

## 功能特性

- **字符级语言模型**：多头自注意力 +  sinusoidal 位置编码 + 解码器层，从零实现（`poem_lm/model.py`）
- **体裁可控**：每句前插入 `<五言>` / `<七言>` 标记 token，模型显式学会控制句长
- **训练优化**：AMP 混合精度（bf16/fp16 自适应）、预热 + 余弦学习率、weight decay 参数分组、梯度裁剪、早停、bf16 优先、`torch.compile`（不可用时自动回退 eager，兼容 Windows 官方 wheel）
- **断点续训**：`--continue` 自动定位最新 checkpoint、反查预设、只补训到目标总轮数（余弦调度正确收尾）
- **条件生成**：给定上句（如"明月松间照"）自动匹配体裁生成下句
- **重复惩罚 / top-k / temperature** 采样控制

## 目录结构

```
prepare_data.py          数据准备：下载全唐诗 -> 解压 -> 繁简转换 -> 构建语料
poem_lm/
    config.py            超参数与预设(small / base / large)
    data.py              词表、token 流数据集、DataLoader
    model.py             多头注意力、位置编码、解码器层、完整模型
    train.py             Trainer(AMP、预热+余弦 LR、验证、checkpoint)
    generate.py          自回归采样与作诗
transformer写短诗.py      命令行入口(训练 / 断点续训 / 作诗)
data/                    数据目录(不入库，见 .gitignore)
checkpoints/             模型权重(不入库)
```

## 快速开始

环境要求：Python 3.10+、PyTorch 2.x，可选 `opencc-python-reimplemented`（繁体转简体，缺失时跳过转换）。

```bash
# 1. 准备数据(只需一次)：下载 chinese-poetry(npm 镜像，国内可达)并构建语料
python prepare_data.py

# 2. 训练(默认 large 预设，约 16M 参数；轮次等超参见 poem_lm/config.py 的 PRESETS)
python transformer写短诗.py

# 3. 只作诗(加载 checkpoints/best.pt)
python transformer写短诗.py --gen-only
```

常用命令：

```bash
python transformer写短诗.py --preset small   # 小模型快速验证流程
python transformer写短诗.py --preset base    # 常规规模
python transformer写短诗.py --continue       # 断点续训(自动找最新 checkpoint)
python transformer写短诗.py --meter 七言      # 全部作七言(五言/七言/auto)
python transformer写短诗.py --prompt 明月松间照   # 条件生成：给上句对下句
python transformer写短诗.py --no-compile     # 关闭 torch.compile(调试/对比用)
```

训练轮次、batch size、学习率等全部写在 `poem_lm/config.py` 的 `PRESETS` 中，直接运行入口即按配置训练，无需命令行传参。

## 模型预设

| 预设 | 参数量 | 结构 | 说明 |
| --- | --- | --- | --- |
| small | ~1M | d_model 128 / 3 层 | 快速验证流程 |
| base | ~7.4M | d_model 256 / 6 层 | 常规训练 |
| large | ~16M | d_model 384 / 8 层 / block 64 / 权重共享 | 突破 base 的 val_loss 平台期 |

large 相对 base 的关键变化：容量翻倍；`block_size` 32→64（一首七言绝句约 40 token，避免截断跨句上下文，让模型学对仗、韵脚）；输出层与词嵌入共享权重（等效正则化）。

## 示例输出

```
--- 第 1 首(七言) ---
春风拂柳绿丝长，
燕子归来入画堂。
独倚阑干看落日，
满山花开满庭香。

--- 条件生成(给上句，试对下句) ---
上句[明月松间照](五言) -> 下句[清泉石上流]
```

## 数据说明

- 数据来源：chinese-poetry npm 包（全唐诗 `poet.tang.*.json`），经 npmmirror 镜像下载
- 过滤：仅保留五言/七言句，按"诗"划分 train/val（同一首诗不跨集合）
- 语料格式：扁平 token 流，每句为 `[<bos> <五言|七言> 句子 <eos>]`
- `data/` 与 `checkpoints/` 体积较大（约 570MB），已在 `.gitignore` 中排除，clone 后运行 `python prepare_data.py` 重新生成

## License

MIT
