#!/usr/bin/env python3
"""tail_punct:尾标点如实化。

问题:归组被 max_dur 上限强制断开时(边界处说话人只停了逗号级的短停顿),
ASR 给段尾打的"。"并不反映真实语调——音频结尾是非终止语调,文本却是句号,
模型会学到"句号有时读成非终止语调"。

修法:利用 merge/input_*.json 里的边界停顿时长,凡"下一组相邻且间隙 <
forced_gap_max(默认 0.8s,与 stage1.split_gap 同量级)"的条目,视为强制
断开,把结尾的"。"改写为","——让标点如实反映非终止语调。只动裸"。",
问号/感叹号/引号收尾一律不动(疑问、感叹语调即使句子未完也多为终止式)。

条目与归组的对应靠"去标点后的纯字符"逐组顺序模糊匹配(相似度 >=0.85):
punct_fix 的 replaced 条目采纳了 align 阶段对导出 wav 的重新转写,个别字
可能与 stage1 转写有出入,不能精确比对;但被 cut 丢弃的组与错位条目的
相似度远低于该阈值,顺序匹配仍是无歧义的,对不上时报错不改任何东西。
就地改写 metadata.csv 与 filelist.txt,变更清单写 tail_punct_report.json。
"""
import difflib
import json
import re

from .config import ProjectConfig

_PUNCT = re.compile(r"[^一-鿿㐀-䶿A-Za-z0-9]")


def strip_punct(t: str) -> str:
    return _PUNCT.sub("", t)


def run(cfg: ProjectConfig, forced_gap_max: float = 0.8):
    meta_path = cfg.dataset_dir / "metadata.csv"
    meta_rows = meta_path.read_text(encoding="utf-8").strip().split("\n")
    meta = [r.split("|", 1) for r in meta_rows]

    # 逐集把 (归组, 边界是否强制断开) 按顺序和 metadata 条目对上
    forced_by_id = {}
    idxs = sorted(p.stem.split("_")[1] for p in cfg.merge_dir.glob("groups_*.json"))
    for idx in idxs:
        rows = json.load(open(cfg.merge_dir / f"input_{idx}.json"))
        groups = json.load(open(cfg.merge_dir / f"groups_{idx}.json"))
        by_i = {r["i"]: r for r in rows}
        ep_meta = [(k, sid, t) for k, (sid, t) in enumerate(meta)
                   if sid.startswith(f"{cfg.seg_prefix}{idx}_")]
        j = 0
        for g in groups:
            gchars = strip_punct("".join(by_i[i]["text"] for i in g))
            if j < len(ep_meta) and difflib.SequenceMatcher(
                    None, strip_punct(ep_meta[j][2]), gchars).ratio() >= 0.85:
                last = by_i[g[-1]]
                forced_by_id[ep_meta[j][1]] = (
                    bool(last.get("next_is_adjacent"))
                    and last.get("gap_to_next") is not None
                    and last["gap_to_next"] < forced_gap_max)
                j += 1
            # 不匹配 = 该组在 cut 阶段被时长过滤丢弃,跳过组、留在原条目
        if j != len(ep_meta):
            raise RuntimeError(
                f"[{idx}] 归组与 metadata 对应失败({j}/{len(ep_meta)}),"
                f"不改任何东西——检查 merge/ 与 dataset/ 是否同代产物。")

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
