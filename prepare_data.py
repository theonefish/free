# -*- coding: utf-8 -*-
"""
全唐诗数据准备脚本
==================
功能：
  1. 下载 chinese-poetry 数据包(npm 镜像，国内可达)，只解压 全唐诗/*.json；
  2. 解析每首诗(作者/诗题/正文)，按标点切句；
  3. 过滤五言/七言句，推断体裁，去重；
  4. 按诗划分 train/val，构建字级词表与扁平 token 流，保存为 .pt 供训练直接加载；
     每句格式为 [<bos> <五言|七言> 句子 <eos>]，体裁标记让模型学会控制句长。

数据字段说明(chinese-poetry JSON)：
    author     : 作者
    title      : 诗题
    paragraphs : 正文(列表，每项一句，带标点)
    id         : 唯一 id
    (朝代统一为"唐"；体裁由句式结构推断，见 infer_genre)

用法：
    python prepare_data.py            # 完整流程(已下载则跳过下载)
    python prepare_data.py --force    # 强制重新下载
"""

import argparse
import io
import json
import re
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import torch

# ---------------------------------------------------------------- 路径配置
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "全唐诗"          # 解压后的原始 JSON
TARBALL = DATA_DIR / "chinese-poetry.tgz"

NPM_TARBALL = "https://registry.npmmirror.com/chinese-poetry/-/chinese-poetry-1.4.1.tgz"
# npm 包内全唐诗文件位于 package/chinese-poetry/json/poet.song.*.json(按千分档编号)
MEMBER_PREFIX = "package/chinese-poetry/json/poet.tang."   # 全唐诗文件(注意:poet.song.* 是宋诗)

# 繁简转换(全唐诗原典为繁体，默认转为简体，更符合现代阅读习惯)
try:
    from opencc import OpenCC
    _T2S = OpenCC("t2s")
except ImportError:
    _T2S = None


def to_simplified(s: str) -> str:
    return _T2S.convert(s) if _T2S is not None else s

OUT_CORPUS_TRAIN = DATA_DIR / "corpus_train.pt"   # 训练 token 流
OUT_CORPUS_VAL = DATA_DIR / "corpus_val.pt"       # 验证 token 流
OUT_VOCAB = DATA_DIR / "vocab.json"               # 词表
OUT_META = DATA_DIR / "meta.json"                 # 统计信息

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<五言>", "<七言>"]
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
# 体裁标记：每句诗前插入，让模型显式知道这句应写几个字
METER_TOKENS = {"五言": "<五言>", "七言": "<七言>"}

PUNCT = "，。！？；：、\n\r\t "


# ---------------------------------------------------------------- 1. 下载与解压
def download_tarball(force=False):
    """下载数据包(带进度显示，已存在且完整则跳过)。"""
    if TARBALL.exists() and not force:
        print(f"[下载] 已存在 {TARBALL.name}({TARBALL.stat().st_size / 1e6:.1f} MB)，跳过")
        return
    print(f"[下载] {NPM_TARBALL}")
    req = urllib.request.Request(NPM_TARBALL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(TARBALL, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = last = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done - last >= 10 * (1 << 20):
                last = done
                print(f"  {done / 1e6:6.1f} MB", flush=True)
    print(f"[下载] 完成 {TARBALL.stat().st_size / 1e6:.1f} MB")


def extract_tangshi():
    """从 tarball 中只解压 全唐诗(poet.tang.*.json)到 data/全唐诗/。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # 清理上一次解压残留的其他诗集(如 poet.song.*.json 是宋诗，不是唐诗)
    for stale in RAW_DIR.glob("*.json"):
        if not stale.name.startswith("poet.tang."):
            stale.unlink()
    existing = list(RAW_DIR.glob("poet.tang.*.json"))
    if existing:
        print(f"[解压] 已存在 {len(existing)} 个全唐诗 JSON，跳过")
        return
    print("[解压] 正在解压 全唐诗(poet.tang.*.json) ...")
    n = 0
    with tarfile.open(TARBALL, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.startswith(MEMBER_PREFIX) and member.name.endswith(".json"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                (RAW_DIR / Path(member.name).name).write_bytes(f.read())
                n += 1
    if n == 0:
        raise RuntimeError("未在数据包中找到 poet.tang.*.json，请检查数据包版本")
    print(f"[解压] 得到 {n} 个文件，共 {sum(p.stat().st_size for p in RAW_DIR.glob('*.json')) / 1e6:.1f} MB")


# ---------------------------------------------------------------- 2. 解析
def clean_sentence(s: str) -> str:
    """去掉首尾空白与标点，返回纯汉字句子。"""
    return s.strip().strip(PUNCT).replace(" ", "")


def infer_genre(sentences):
    """根据句数与每句字数推断体裁。"""
    if not sentences:
        return "未知"
    lens = {len(s) for s in sentences}
    n = len(sentences)
    if lens == {5}:
        return "五言绝句" if n == 4 else "五言律诗" if n == 8 else "五言古诗"
    if lens == {7}:
        return "七言绝句" if n == 4 else "七言律诗" if n == 8 else "七言古诗"
    if lens <= {5, 7}:
        return "五七言古诗"
    return "杂言古诗"


def load_poems(simplify=True):
    """读取全部 poet.tang.*.json，返回 [(author, title, [sentence...], genre), ...]"""
    poems = []
    files = sorted(RAW_DIR.glob("poet.tang.*.json"))
    if not files:
        raise FileNotFoundError(f"{RAW_DIR} 中没有数据文件，请先运行下载")
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [警告] {path.name} 解析失败: {e}")
            continue
        for item in data:
            sents = []
            for para in item.get("paragraphs", []):
                for piece in re.split(r"[，。！？；：、]", para):
                    piece = clean_sentence(piece)
                    if piece and all("\u4e00" <= c <= "\u9fff" for c in piece):
                        sents.append(to_simplified(piece) if simplify else piece)
            if sents:
                poems.append((to_simplified(item.get("author", "佚名")) if simplify
                              else item.get("author", "佚名"),
                              to_simplified(item.get("title", "无题")) if simplify
                              else item.get("title", "无题"),
                              sents,
                              infer_genre(sents)))
    return poems


# ---------------------------------------------------------------- 3. 构建语料
def build_corpus(poems, min_freq=1, val_ratio=0.02):
    """
    构建训练语料：
      - 只保留五言/七言句(模型专注这两种句式)；
      - 去重；
      - 按"诗"划分 train/val(同一首诗不跨集合，避免泄题)；
      - 字级词表(低于 min_freq 的字映射为 <unk>)；
      - 扁平 token 流：[<bos> 字... <eos>] 首尾相接，训练时随机切窗口。
    """
    train_lines, val_lines = [], []
    genre_counter, author_counter = Counter(), Counter()
    seen = set()
    n_poem = 0
    for author, title, sents, genre in poems:
        author_counter[author] += 1
        genre_counter[genre] += 1
        n_poem += 1
        keep = [s for s in sents if len(s) in (5, 7)]
        if not keep:
            continue
        # 用 (作者, 诗题, 首句) 去重
        key = (author, title, keep[0])
        if key in seen:
            continue
        seen.add(key)
        # 每 1/val_ratio 首诗进验证集(近似比例，确定性切分)
        (val_lines if (n_poem % max(1, round(1 / val_ratio))) == 0 else train_lines).extend(keep)

    # 词表
    freq = Counter("".join(train_lines))
    chars = sorted(c for c, n in freq.items() if n >= min_freq)
    itos = SPECIAL_TOKENS + chars
    stoi = {c: i for i, c in enumerate(itos)}

    def to_stream(lines):
        ids = []
        for line in lines:
            ids.append(BOS_IDX)
            # 体裁标记：<五言>/<七言>，按句子实际字数选择
            meter = {5: "五言", 7: "七言"}[len(line)]
            ids.append(stoi[METER_TOKENS[meter]])
            ids.extend(stoi.get(c, UNK_IDX) for c in line)
            ids.append(EOS_IDX)
        return torch.tensor(ids, dtype=torch.int32)

    stats = {
        "poems_total": n_poem,
        "train_sentences": len(train_lines),
        "val_sentences": len(val_lines),
        "vocab_size": len(itos),
        "train_tokens": None,  # 下面填
        "val_tokens": None,
        "genres": dict(genre_counter.most_common()),
        "top_authors": dict(author_counter.most_common(20)),
        "rare_chars_as_unk": sum(n for c, n in freq.items() if n < min_freq),
    }
    return to_stream(train_lines), to_stream(val_lines), itos, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="强制重新下载")
    ap.add_argument("--min-freq", type=int, default=2, help="词表最小字频")
    ap.add_argument("--keep-traditional", action="store_true",
                    help="保留繁体(默认转换为简体)")
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    download_tarball(force=args.force)
    extract_tangshi()

    print("[解析] 读取全唐诗 ...")
    poems = load_poems(simplify=not args.keep_traditional)
    print(f"[解析] 共 {len(poems)} 首诗")

    print("[构建] 生成词表与 token 流 ...")
    train_stream, val_stream, itos, stats = build_corpus(poems, min_freq=args.min_freq)
    stats["train_tokens"] = train_stream.numel()
    stats["val_tokens"] = val_stream.numel()

    torch.save(train_stream, OUT_CORPUS_TRAIN)
    torch.save(val_stream, OUT_CORPUS_VAL)
    OUT_VOCAB.write_text(json.dumps(itos, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_META.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n========== 数据统计 ==========")
    print(f"诗歌总数      : {stats['poems_total']}")
    print(f"训练句子数    : {stats['train_sentences']}")
    print(f"验证句子数    : {stats['val_sentences']}")
    print(f"词表大小      : {stats['vocab_size']}")
    print(f"训练 token 数 : {stats['train_tokens']}")
    print(f"验证 token 数 : {stats['val_tokens']}")
    print(f"体裁分布      : {stats['genres']}")
    print(f"出现最多作者  : {list(stats['top_authors'])[:10]}")
    print(f"\n已保存:\n  {OUT_CORPUS_TRAIN}\n  {OUT_CORPUS_VAL}\n  {OUT_VOCAB}\n  {OUT_META}")


if __name__ == "__main__":
    main()
