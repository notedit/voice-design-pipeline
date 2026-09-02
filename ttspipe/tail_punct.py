#!/usr/bin/env python3
"""tail_punct:尾标点如实化。

问题:归组被 max_dur 上限强制断开时(边界处说话人只停了逗号级的短停顿),
ASR 给段尾打的"。"并不反映真实语调——音频结尾是非终止语调,文本却是句号,
模型会学到"句号有时读成非终止语调"。

修法:利用 merge/input_*.json 里的边界停顿时长,凡"下一组相邻且间隙 <
forced_gap_max(默认 0.8s,与 stage1.split_gap 同量级)"的条目,视为强制
断开,把结尾的"。"改写为","——让标点如实反映非终止语调。只动裸"。",
问号/感叹号/引号收尾一律不动(疑问、感叹语调即使句子未完也多为终止式)。

条目与归组的对应优先读 dataset/manifest.json(export 按构造写下每条来自
哪集哪组,零猜测);没有 manifest 的旧数据集才退回"去标点纯字符"顺序模糊
匹配(fix-punct 采纳整条重转写后,个别条目与段级文本差异可超 15%,模糊
匹配会失去同步——这正是改用 manifest 的原因),对不上时报错不改任何东西。
就地改写 metadata.csv 与 filelist.txt,变更清单写 tail_punct_report.json。
"""
import difflib
import json
import re

from .config import ProjectConfig
from .provenance import files_hash, require_fresh

_PUNCT = re.compile(r"[^一-鿿㐀-䶿A-Za-z0-9]")


def strip_punct(t: str) -> str:
    return _PUNCT.sub("", t)


def pair_groups(ep_texts, groups, by_i, forced_gap_max=0.8, min_ratio=0.85):
    """纯函数:把本集 metadata 文本(按导出顺序)与归组顺序对上,返回每条
    是否"强制断开"的列表(长度 = len(ep_texts));对不上抛 ValueError。"""
    out, j = [], 0
    for g in groups:
        gchars = strip_punct("".join(by_i[i]["text"] for i in g))
        if j < len(ep_texts) and difflib.SequenceMatcher(
                None, strip_punct(ep_texts[j]), gchars).ratio() >= min_ratio:
            last = by_i[g[-1]]
            out.append(bool(last.get("next_is_adjacent"))
                       and last.get("gap_to_next") is not None
                       and last["gap_to_next"] < forced_gap_max)
            j += 1
    if j != len(ep_texts):
        raise ValueError(f"归组与 metadata 对应失败({j}/{len(ep_texts)})")
    return out


def run(cfg: ProjectConfig, forced_gap_max: float = 0.8):
    meta_path = cfg.dataset_dir / "metadata.csv"
    meta_rows = meta_path.read_text(encoding="utf-8").strip().split("\n")
    meta = [r.split("|", 1) for r in meta_rows]
    require_fresh(cfg.out_path, "export",
                  {"merge": files_hash(cfg.merge_dir.glob("groups_*.json"))},
                  consumer="fix-tail-punct")

    forced_by_id = {}
    idxs = sorted(p.stem.split("_")[1] for p in cfg.merge_dir.glob("groups_*.json"))
    manifest_path = cfg.dataset_dir / "manifest.json"
    manifest = json.load(open(manifest_path, encoding="utf-8")) if manifest_path.exists() else None
    for idx in idxs:
        rows = json.load(open(cfg.merge_dir / f"input_{idx}.json"))
        groups = json.load(open(cfg.merge_dir / f"groups_{idx}.json"))
        by_i = {r["i"]: r for r in rows}
        ep_meta = [(k, sid, t) for k, (sid, t) in enumerate(meta)
                   if sid.startswith(f"{cfg.seg_prefix}{idx}_")]
        if manifest is not None:
            # 按构造对应:manifest 记的 segs 与 groups 一致才算同代
            for _, sid, _ in ep_meta:
                m = manifest.get(sid)
                if m is None or m["gid"] >= len(groups) or groups[m["gid"]] != m["segs"]:
                    raise RuntimeError(f"[{idx}] manifest 与 merge/groups 不一致({sid}),"
                                       f"不改任何东西——merge/ 与 dataset/ 不是同代产物。")
                last = by_i[m["segs"][-1]]
                forced_by_id[sid] = (bool(last.get("next_is_adjacent"))
                                     and last.get("gap_to_next") is not None
                                     and last["gap_to_next"] < forced_gap_max)
            continue
        try:   # 旧数据集无 manifest:退回模糊匹配
            flags = pair_groups([t for _, _, t in ep_meta], groups, by_i, forced_gap_max)
        except ValueError as e:
            raise RuntimeError(f"[{idx}] {e},不改任何东西——检查 merge/ 与 "
                               f"dataset/ 是否同代产物。") from e
        for (_, sid, _), f in zip(ep_meta, flags):
            forced_by_id[sid] = f

    changed = []
    for k, (sid, text) in enumerate(meta):
        if forced_by_id.get(sid) and text.rstrip().endswith("。"):
            meta[k][1] = text.rstrip()[:-1] + ","
            changed.append(sid)

    if changed:
        meta_path.write_text("\n".join(f"{sid}|{t}" for sid, t in meta) + "\n",
                             encoding="utf-8")
        fl_path = cfg.dataset_dir / "filelist.txt"
        by_id = dict(meta)
        fl_rows = []
        for line in fl_path.read_text(encoding="utf-8").strip().split("\n"):
            wav, spk, lang, _ = line.split("|", 3)
            sid = wav.split("/")[-1].removesuffix(".wav")
            fl_rows.append(f"{wav}|{spk}|{lang}|{by_id[sid]}")
        fl_path.write_text("\n".join(fl_rows) + "\n", encoding="utf-8")

    n_forced = sum(1 for v in forced_by_id.values() if v)
    report = dict(forced_gap_max=forced_gap_max, n_utts=len(meta),
                  n_forced_boundary=n_forced, n_rewritten=len(changed),
                  rewritten=changed)
    rep_path = cfg.out_path / "tail_punct_report.json"
    json.dump(report, open(rep_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"TAIL-PUNCT DONE forced_boundary={n_forced}/{len(meta)} "
          f"rewritten(。->,)={len(changed)} -> {rep_path}", flush=True)
    return report
