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

已知限制(有意不修,详见 README):文本匹配是逐字符顺序指针walk,没有
重新同步能力——新旧文本在靠前的位置一旦有一处标点/字符对不上,后面
全部错位,覆盖率跌破 80% 阈值,整条退回旧文本不做任何校正。长句(20s+)
+ 口语化内容(重复、自我修正)命中率明显更高:kuaidao 348 条里 191 条因
此没被校正(kept_old),而不是文本对不上导致的转写失败。修这个需要换成
真正的 diff/对齐算法,而不是指针 walk——留作后续工作,不在这次重构范围内,
因为修复会改变所有项目的输出,不是纯粹的模块化。
"""
import difflib
import json

from .config import ProjectConfig

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
        # 对齐匹配(逐字符顺序指针,见模块 docstring 的已知限制)
        ptr = 0
        seq = []
        for ch in text:
            if ptr < len(items) and items[ptr][0] == ch:
                seq.append((ch, items[ptr]))
                ptr += 1
            else:
                seq.append((ch, None))
        if ptr < len(items) * 0.8:
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
        if src.exists() and not dst.exists():
            src.rename(dst)

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
