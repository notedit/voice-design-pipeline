#!/usr/bin/env python3
"""从 report_XXX.json 生成归组输入 + 贪心归组基线。对应旧的 prep_merge.py +
prep_review.py——6 个项目里这两步完全没有分叉,机械搬过来,常量从
ProjectConfig.merge_review 读(6 个项目实际都是 2.5/30.0,从没变过)。

prep_input(): report -> merge/input_XXX.json(逐段 + 到下一段的间隙)
greedy_baseline(): input -> merge/greedy_XXX.json(贪心归组基线) +
                    merge/review_XXX.json(多段组语义拆分复审用,可选)
"""
import json

from .config import ProjectConfig


def prep_input(cfg: ProjectConfig):
    cfg.merge_dir.mkdir(exist_ok=True, parents=True)
    reports = [json.load(open(p)) for p in sorted(cfg.reports_dir.glob("report_*.json"))]
    for ep in reports:
        idx = ep["episode"]
        kept = [s for s in ep["segments"] if s["kept"]]
        rows = []
        for j, s in enumerate(kept):
            nxt = kept[j + 1] if j + 1 < len(kept) else None
            rows.append({
                "i": s["i"],
                "start": s["start"], "end": s["end"],
                "dur": round(s["end"] - s["start"], 2),
                "gap_to_next": round(nxt["start"] - s["end"], 2) if nxt else None,
                "next_is_adjacent": bool(nxt and nxt["i"] == s["i"] + 1),
                "text": s["text"],
            })
        json.dump(rows, open(cfg.merge_dir / f"input_{idx}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(idx, len(rows))


def greedy_baseline(cfg: ProjectConfig):
    max_gap, max_dur = cfg.merge_review.max_gap, cfg.merge_review.max_dur
    n_groups = n_multi = 0
    for f in sorted(cfg.merge_dir.glob("input_*.json")):
        idx = f.stem.split("_")[1]
        rows = json.load(open(f))
        by_i = {r["i"]: r for r in rows}
        groups, cur = [], []
        for r in rows:
            if cur:
                prev = by_i[cur[-1]]
                new_dur = r["end"] - by_i[cur[0]]["start"]
                if prev["next_is_adjacent"] and prev["gap_to_next"] <= max_gap \
                        and new_dur <= max_dur:
                    cur.append(r["i"])
                    continue
                groups.append(cur)
                cur = [r["i"]]
            else:
                cur = [r["i"]]
        if cur:
            groups.append(cur)
        json.dump(groups, open(cfg.merge_dir / f"greedy_{idx}.json", "w"))
        review = []
        for gid, g in enumerate(groups):
            if len(g) < 2:
                continue
            review.append({
                "gid": gid,
                "dur": round(by_i[g[-1]]["end"] - by_i[g[0]]["start"], 1),
                "parts": [{"i": i,
                           "gap_after": by_i[i]["gap_to_next"] if i != g[-1] else None,
                           "text": by_i[i]["text"]} for i in g],
            })
        json.dump(review, open(cfg.merge_dir / f"review_{idx}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        n_groups += len(groups)
        n_multi += len(review)
    print(f"episodes={len(list(cfg.merge_dir.glob('greedy_*.json')))} "
          f"groups={n_groups} multi-seg groups to review={n_multi}")


def run(cfg: ProjectConfig):
    prep_input(cfg)
    greedy_baseline(cfg)


if __name__ == "__main__":
    import sys
    from .config import load_project
    run(load_project(sys.argv[1]))
