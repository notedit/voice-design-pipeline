#!/usr/bin/env python3
"""静音异常检测(强制对齐时间戳法):不分析波形,只用 alignments.jsonl 的字级
时间戳。对应旧的各项目 qa_silence_outliers.py。

- 句首静音 = 第一个对齐字的开始时间;句尾静音 = 音频总时长 - 最后一个对齐字
  的结束时间。这两组用 IQR(Q3+1.5*IQR)定异常边界——数据驱动,不是拍脑袋
  阈值,每个项目按自己的分布重新算。
- 句中/句末停顿 = 相邻两个对齐字之间的时间差,按夹在中间的标点分类(句中/
  句末/无)。这组因为绝大多数字间距接近 0(正常连续发音),IQR 在这种分布
  上不适用,用固定阈值(ProjectConfig.qa.fixed_gap_th)。这个阈值必须跟着
  cut 阶段的 max_gap/gap_pass_ratio 走:kuaidao/wowan 把 max_gap 从 0.8 松到
  1.2 并改比例压缩后,合并边界的自然停顿设计上能到 ~1.66s,旧的 1.15s 固定
  阈值会把刻意保留的长尾也判成异常,所以这两个项目配了 1.5;还在用旧 cut
  参数的项目仍配 1.15,两者要对应,不能混着用。

match() 用 token 级指针匹配,不是逐字符——这是本模块相对旧脚本唯一修的
真 bug:强制对齐对英文/数字串(比如 "SUV")会整体输出成一个多字符 item,
逐字符比较会让这类 item 永远匹配不上,把它前面的字误判成"最后一个对齐字",
拼出虚假的句尾静音。所有项目统一吃这个修复,不放进配置。
"""
import argparse
import difflib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import ProjectConfig, load_project
from .provenance import dataset_fingerprint, require_fresh

MID = set("，,、；;：:—")
FIN = set("。！？!?…~～.")


def load_alignments(cfg: ProjectConfig):
    aligns = {}
    for line in open(cfg.alignments_path, encoding="utf-8"):
        d = json.loads(line)
        aligns[d["id"]] = d
    return aligns


def match(text, items):
    """把 text 对齐到 items(token 级指针匹配),返回 (seq, ptr)。
    seq 每项是 (token_str, item_or_None)。"""
    i = ptr = 0
    seq = []
    while i < len(text):
        if ptr < len(items):
            tok = items[ptr][0]
            if text[i:i + len(tok)] == tok:
                seq.append((tok, items[ptr]))
                i += len(tok)
                ptr += 1
                continue
        seq.append((text[i], None))
        i += 1
    return seq, ptr


def analyze(dataset_dir: Path, aligns: dict):
    meta = dict(l.split("|", 1) for l in
                open(dataset_dir / "metadata.csv").read().strip().split("\n"))
    rows = []
    for sid in sorted(meta):
        text = meta[sid]
        d = aligns.get(sid)
        if d is None:
            continue
        sim = difflib.SequenceMatcher(None, text, d["text"]).ratio()
        seq, ptr = match(text, d["items"])
        if sim < 0.5 or ptr < len(d["items"]) * 0.8:
            continue
        aligned = [(i, ch, it) for i, (ch, it) in enumerate(seq) if it is not None]
        if len(aligned) < 2:
            continue
        dur = sf.info(dataset_dir / "wavs" / f"{sid}.wav").duration
        head = aligned[0][2][1]
        tail = dur - aligned[-1][2][2]
        gaps = []
        for k in range(len(aligned) - 1):
            i0, _, it0 = aligned[k]
            i1, _, it1 = aligned[k + 1]
            gap = it1[1] - it0[2]
            between = [seq[j][0] for j in range(i0 + 1, i1)]
            if any(c in FIN for c in between):
                btype = "fin"
            elif any(c in MID for c in between):
                btype = "mid"
            else:
                btype = "none"
            gaps.append((btype, gap, i0, i1))
        rows.append(dict(id=sid, dur=dur, head=head, tail=tail, gaps=gaps))
    return rows


def iqr_bound(arr):
    q1, q3 = np.percentile(arr, [25, 75])
    return float(q3 + 1.5 * (q3 - q1))


def flag(rows, fixed_gap_th):
    head_arr = np.array([r["head"] for r in rows])
    tail_arr = np.array([r["tail"] for r in rows])
    head_bound = iqr_bound(head_arr)
    tail_bound = iqr_bound(tail_arr)
    flagged = {}
    for r in rows:
        reasons = []
        if r["head"] > head_bound:
            reasons.append(f"head={r['head']:.2f}s>{head_bound:.2f}")
        if r["tail"] > tail_bound:
            reasons.append(f"tail={r['tail']:.2f}s>{tail_bound:.2f}")
        for btype, gap, i0, i1 in r["gaps"]:
            if btype in ("mid", "fin") and gap >= fixed_gap_th:
                reasons.append(f"{btype}_gap={gap:.2f}s@{i0}-{i1}")
        if reasons:
            flagged[r["id"]] = reasons
    return flagged, head_bound, tail_bound


def distribution_stats(rows):
    head = np.array([r["head"] for r in rows])
    tail = np.array([r["tail"] for r in rows])
    mid, fin, none = [], [], []
    for r in rows:
        for btype, gap, _, _ in r["gaps"]:
            {"mid": mid, "fin": fin, "none": none}[btype].append(gap)
    mid, fin, none = map(np.array, (mid, fin, none))

    def stat(arr):
        if len(arr) == 0:
            return dict(n=0, mean=None, median=None, p90=None, max=None)
        return dict(n=len(arr), mean=round(float(arr.mean()), 3),
                    median=round(float(np.median(arr)), 3),
                    p90=round(float(np.percentile(arr, 90)), 3),
                    max=round(float(arr.max()), 3))

    return {"head": stat(head), "tail": stat(tail), "mid_gap": stat(mid),
            "fin_gap": stat(fin), "none_gap": stat(none)}


def apply_removal(dataset_dir: Path, flagged: dict, speaker_tag: str):
    for name, bak in (("metadata.csv", "metadata_presilence.csv"),
                      ("filelist.txt", "filelist_presilence.txt")):
        src, dst = dataset_dir / name, dataset_dir / bak
        if src.exists() and not dst.exists():
            src.rename(dst)
    old_meta = dict(l.split("|", 1) for l in
                    open(dataset_dir / "metadata_presilence.csv").read().strip().split("\n"))
    keep_ids = [sid for sid in sorted(old_meta) if sid not in flagged]
    meta_rows = [f"{sid}|{old_meta[sid]}" for sid in keep_ids]
    fl_rows = [f"wavs/{sid}.wav|{speaker_tag}|ZH|{old_meta[sid]}" for sid in keep_ids]
    (dataset_dir / "metadata.csv").write_text("\n".join(meta_rows) + "\n", encoding="utf-8")
    (dataset_dir / "filelist.txt").write_text("\n".join(fl_rows) + "\n", encoding="utf-8")
    n_removed = 0
    for sid in flagged:
        wav = dataset_dir / "wavs" / f"{sid}.wav"
        if wav.exists():
            wav.unlink()
            n_removed += 1
    return len(keep_ids), n_removed


def run(cfg: ProjectConfig, dataset_dir: Path = None, apply: bool = False):
    require_fresh(cfg.out_path, "align",
                  {"dataset": dataset_fingerprint(dataset_dir or cfg.dataset_dir)},
                  consumer="qa-silence")
    dataset_dir = dataset_dir or cfg.dataset_dir
    aligns = load_alignments(cfg)
    rows = analyze(dataset_dir, aligns)
    flagged, head_bound, tail_bound = flag(rows, cfg.qa.fixed_gap_th)
    dist = distribution_stats(rows)

    report = {"dataset_dir": str(dataset_dir), "n_total": len(rows),
              "head_bound": round(head_bound, 3), "tail_bound": round(tail_bound, 3),
              "fixed_gap_threshold": cfg.qa.fixed_gap_th,
              "n_flagged": len(flagged), "flagged": flagged, "distribution": dist}
    out_path = cfg.out_path / f"qa_silence_outliers_{dataset_dir.name}.json"
    json.dump(report, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{dataset_dir.name}] total={len(rows)} flagged={len(flagged)} "
          f"head_bound={head_bound:.2f} tail_bound={tail_bound:.2f} -> {out_path}")

    if apply and flagged:
        n_kept, n_removed = apply_removal(dataset_dir, flagged, cfg.speaker_tag)
        print(f"removed {n_removed} wavs, kept {n_kept}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("dataset_dir", nargs="?", default=None)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    cfg = load_project(args.project)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    run(cfg, dataset_dir=dataset_dir, apply=args.apply)


if __name__ == "__main__":
    main()
