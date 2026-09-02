#!/usr/bin/env python3
"""按强制对齐的实测停顿校正标点,重写 metadata/filelist。对应旧的 punct_fix.py,
机械搬过来、行为完全不变。

规则(阈值来自当年 qa_punct_stats.json 的分布,6 个项目一直共用,没有配置化):
- 句中标点(，,、;;::—)处实测停顿 < 0.10s -> 删除该标点(假逗号)
- 无标点字间停顿 > 0.50s -> 补「,」
- 句末标点(。!?…~)一律保留
文本整体替换为全句重转写版本(跨段标点更自然);与旧文本相似度 < 0.5 的
条目视为转写异常,保留旧文本不做校正。
旧版备份:metadata_prepunct.csv / filelist_prepunct.txt

文本 <-> 时间戳的匹配用 textalign.match_chars(difflib,可重新同步):旧的
逐字符指针 walk 遇到多字符 token(英文/数字)或规范化差异就整条错位,
覆盖率跌破 80% 后退回旧文本——kuaidao 一度 37% 条目因此没被校正。换成
diff 对齐后覆盖率阈值(0.8)不变,只是能对上的条目多了。
文本权威:能对上的条目一律采用 align 阶段对整条音频的重新转写(有完整
上下文),旧的段级拼接文本只用于相似度守门。
"""
import difflib
import json

from .config import ProjectConfig
from .provenance import dataset_fingerprint, require_fresh
from .textalign import match_chars

MID = set("，,、；;：:—")
FIN = set("。！？!?…~～.")
DEL_TH = 0.10
INS_TH = 0.50


def has_chinese(t):
    return any("一" <= c <= "鿿" for c in t)


def run(cfg: ProjectConfig):
    dataset_dir = cfg.dataset_dir
    old_meta = dict(l.split("|", 1) for l in
                    open(dataset_dir / "metadata.csv").read().strip().split("\n"))

    require_fresh(cfg.out_path, "align",
                  {"dataset": dataset_fingerprint(dataset_dir)}, consumer="fix-punct")
    aligns = {}
    for line in open(cfg.alignments_path, encoding="utf-8"):
        d = json.loads(line)
        aligns[d["id"]] = d

    n_replaced = n_kept_old = n_del = n_ins = n_lowsim = 0
    samples = []
    new_meta = {}
    for sid in sorted(old_meta):
        old_text = old_meta[sid]
        d = aligns.get(sid)
        if d is None:
            new_meta[sid] = old_text
            n_kept_old += 1
            continue
        text, items = d["text"], d["items"]
        sim = difflib.SequenceMatcher(None, old_text, text).ratio()
        if sim < 0.5 or len(text) < 2 or not has_chinese(text):
            new_meta[sid] = old_text
            n_lowsim += 1
            continue
        # difflib 对齐(可重新同步),覆盖率 <0.8 仍视为对不上,退回旧文本
        seq, coverage = match_chars(text, items)
        if coverage < 0.8:
            new_meta[sid] = old_text
            n_kept_old += 1
            continue
        # 逐边界改写
        out = []
        j = 0
        deleted = inserted = 0
        while j < len(seq):
            ch, it = seq[j]
            out.append(ch)
            if it is not None:
                k = j + 1
                tail = []
                while k < len(seq) and seq[k][1] is None:
                    tail.append(seq[k][0])
                    k += 1
                if k < len(seq):
                    gap = seq[k][1][1] - it[2]
                    mids = [c for c in tail if c in MID]
                    fins = [c for c in tail if c in FIN]
                    others = [c for c in tail if c not in MID and c not in FIN]
                    if mids and not fins and gap < DEL_TH:
                        tail = others
                        deleted += len(mids)
                    elif not mids and not fins and gap > INS_TH:
                        tail = others + ["，"]
                        inserted += 1
                    out.extend(tail)
                    j = k
                    continue
                else:
                    out.extend(tail)
                    j = k
                    continue
            j += 1
        fixed = "".join(out)
        n_del += deleted
        n_ins += inserted
        if (deleted or inserted) and len(samples) < 12:
            samples.append((sid, old_text, fixed))
        new_meta[sid] = fixed
        n_replaced += 1

    for name, bak in (("metadata.csv", "metadata_prepunct.csv"),
                      ("filelist.txt", "filelist_prepunct.txt")):
        src, dst = dataset_dir / name, dataset_dir / bak
        if src.exists():
            src.replace(dst)   # 总是覆盖:备份必须与当前 dataset 同代

    meta_rows = [f"{sid}|{new_meta[sid]}" for sid in sorted(new_meta)]
    fl_rows = [f"wavs/{sid}.wav|{cfg.speaker_tag}|ZH|{new_meta[sid]}" for sid in sorted(new_meta)]
    (dataset_dir / "metadata.csv").write_text("\n".join(meta_rows) + "\n", encoding="utf-8")
    (dataset_dir / "filelist.txt").write_text("\n".join(fl_rows) + "\n", encoding="utf-8")

    print(f"replaced={n_replaced} kept_old(no-align/match-fail)={n_kept_old} "
          f"lowsim_kept_old={n_lowsim}")
    print(f"fake commas deleted={n_del}, missing pauses inserted={n_ins}")
    print("\n--- 样例(前12条有改动的) ---")
    for sid, a, b in samples:
        print(f"[{sid}]\n  旧: {a[:70]}\n  新: {b[:70]}")


if __name__ == "__main__":
    import sys
    from .config import load_project
    run(load_project(sys.argv[1]))
