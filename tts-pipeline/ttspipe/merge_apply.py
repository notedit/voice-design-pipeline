#!/usr/bin/env python3
"""把语义拆分结果应用到贪心基线,产出最终 merge/groups_XXX.json,再校验。
对应旧的 apply_splits.py + validate_groups.py,机械搬过来。

apply_splits(): 如果 merge/ 下没有 splits_XXX.json / splits2_batch*.json
(语义拆分复审的产出——来自 Claude Code 会话内子代理或 merge_llm.py 的
API 调用),贪心基线直接原样生效,流水线不会卡住。
validate(): 校验 groups 覆盖完整、合并条件满足、组时长不超限。
"""
import json
import sys

from .config import ProjectConfig


def apply_splits(cfg: ProjectConfig):
    merge_dir = cfg.merge_dir
    splits2 = {}
    for f in merge_dir.glob("splits2_batch*.json"):
        for item in json.load(open(f)):
            splits2[(item["ep"], item["gid"])] = item["groups"]

    total = split_cnt = 0
    for f in sorted(merge_dir.glob("greedy_*.json")):
        idx = f.stem.split("_")[1]
        greedy = json.load(open(f))
        sp_path = merge_dir / f"splits_{idx}.json"
        splits = {}
        if sp_path.exists():
            for item in json.load(open(sp_path)):
                splits[item["gid"]] = item["groups"]
        for (ep, gid), groups in splits2.items():
            if ep == idx:
                splits[gid] = groups  # 第二轮复审优先
        out = []
        for gid, g in enumerate(greedy):
            if gid in splits:
                sub = splits[gid]
                flat = [i for s in sub for i in s]
                if flat != g or any(not s for s in sub):
                    print(f"[{idx}] gid={gid} 非法拆分(不是原组的连续划分),保留原组")
                    out.append(g)
                else:
                    out.extend(sub)
                    split_cnt += 1
            else:
                out.append(g)
        json.dump(out, open(merge_dir / f"groups_{idx}.json", "w"))
        total += 1
    print(f"applied: {total} episodes, {split_cnt} groups split")


def validate(cfg: ProjectConfig, idxs=None) -> bool:
    merge_dir = cfg.merge_dir
    max_gap, max_dur = cfg.merge_review.max_gap, cfg.merge_review.max_dur
    if idxs is None:
        idxs = sorted(f.stem.split("_")[1] for f in merge_dir.glob("input_*.json"))
    fail = False
    for idx in idxs:
        rows = json.load(open(merge_dir / f"input_{idx}.json"))
        try:
            groups = json.load(open(merge_dir / f"groups_{idx}.json"))
        except Exception as e:
            print(f"[{idx}] FAIL: 无法读取 groups 文件: {e}")
            fail = True
            continue
        by_i = {r["i"]: r for r in rows}
        order = [r["i"] for r in rows]
        flat = [i for g in groups for i in g]
        err = None
        if sorted(flat) != sorted(order):
            err = "覆盖不完整或有重复/多余的 i"
        elif flat != order:
            err = "组内或组间顺序与原列表不一致"
        else:
            for g in groups:
                for a in g[:-1]:
                    ra = by_i[a]
                    if not ra["next_is_adjacent"] or ra["gap_to_next"] > max_gap:
                        err = f"i={a} 不满足合并条件"
                        break
                if err:
                    break
                dur = by_i[g[-1]]["end"] - by_i[g[0]]["start"]
                if dur > max_dur:
                    err = f"组 {g} 时长 {dur:.1f}s 超限"
                    break
        if err:
            print(f"[{idx}] FAIL: {err}")
            fail = True
        else:
            print(f"[{idx}] OK ({len(groups)} groups)")
    print("ALL OK" if not fail else "HAS FAILURES")
    return not fail


def run(cfg: ProjectConfig):
    apply_splits(cfg)
    ok = validate(cfg)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    from .config import load_project
    run(load_project(sys.argv[1]))
