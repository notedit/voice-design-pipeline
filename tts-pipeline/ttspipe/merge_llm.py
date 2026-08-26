#!/usr/bin/env python3
"""用 Claude Haiku 4.5 做相邻分段的语义归组(替代会话内子代理,可全自动复用)。
对应旧的 merge_llm.py,机械搬过来,常量从 ProjectConfig.merge_review 读。

输入:  merge/input_XXX.json  (merge_prep.prep_input 生成)
输出:  merge/groups_XXX.json (与会话内子代理产出同格式,之后跑 cut.run)

用法: python -m ttspipe.merge_llm <project> [ep_idx ...]
需要 ANTHROPIC_API_KEY。
"""
import json
import sys

from .config import ProjectConfig

MODEL = "claude-haiku-4-5"

GROUPS_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}

PROMPT = """下面是一个按时间排序的中文语音分段列表(JSON),每项有 i(原始索引)、start/end/dur(秒)、gap_to_next(与下一段的间隙秒数)、next_is_adjacent(下一段原始索引是否紧邻)、text(该段文本)。

任务:把相邻分段归组为语义完整的话语单元,用于 TTS 训练。规则:
1. 组内必须是列表中连续的行;只有当前一行 next_is_adjacent 为 true 且 gap_to_next <= {max_gap} 时,才允许把下一行并入同组。
2. 组的总时长(组内最后一段 end - 第一段 start)不得超过 {max_dur} 秒,理想为 2~15 秒。
3. 合并依据语义:未说完的句子、悬垂连词、紧密衔接的上下句应合并;话题转换、场景切换处必须断开。
4. 允许单段成组。每个 i 必须恰好出现在一个组里,不得遗漏或重排。

输出 JSON:{{"groups": [[组1的i值...], [组2的i值...], ...]}},组按时间顺序排列。

分段列表:
"""


def validate(groups, rows, max_gap, max_dur):
    by_i = {r["i"]: r for r in rows}
    order = [r["i"] for r in rows]
    flat = [i for g in groups for i in g]
    if sorted(flat) != sorted(order):
        return "覆盖不完整或有重复/多余的 i"
    if flat != order:
        return "组内或组间顺序与原列表不一致"
    for g in groups:
        for a, b in zip(g, g[1:]):
            ra = by_i[a]
            if not ra["next_is_adjacent"] or ra["gap_to_next"] > max_gap:
                return f"i={a}->{b} 不满足合并条件 (adjacent={ra['next_is_adjacent']}, gap={ra['gap_to_next']})"
        dur = by_i[g[-1]]["end"] - by_i[g[0]]["start"]
        if dur > max_dur:
            return f"组 {g} 时长 {dur:.1f}s 超过 {max_dur}s"
    return None


def merge_episode(client, merge_dir, idx, max_gap, max_dur):
    rows = json.load(open(merge_dir / f"input_{idx}.json"))
    prompt = PROMPT.format(max_gap=max_gap, max_dur=max_dur)
    messages = [{"role": "user", "content": prompt + json.dumps(rows, ensure_ascii=False)}]
    for attempt in range(3):
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            output_config={"format": {"type": "json_schema", "schema": GROUPS_SCHEMA}},
            messages=messages,
        )
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"[{idx}] 输出被截断,需提高 max_tokens")
        text = next(b.text for b in response.content if b.type == "text")
        groups = json.loads(text)["groups"]
        err = validate(groups, rows, max_gap, max_dur)
        if err is None:
            return groups
        print(f"[{idx}] 第{attempt + 1}次校验失败: {err},重试", flush=True)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content":
                         f"你的分组未通过校验:{err}。请修正后重新输出完整的 groups JSON,遵守全部规则。"})
    print(f"[{idx}] LLM 归组多次失败,回退为逐段成组", flush=True)
    return [[r["i"]] for r in rows]


def run(cfg: ProjectConfig, only=None):
    import anthropic
    client = anthropic.Anthropic()
    max_gap, max_dur = cfg.merge_review.max_gap, cfg.merge_review.max_dur
    only = set(only) if only else None
    inputs = sorted(cfg.merge_dir.glob("input_*.json"))
    for f in inputs:
        idx = f.stem.split("_")[1]
        if only and idx not in only:
            continue
        groups = merge_episode(client, cfg.merge_dir, idx, max_gap, max_dur)
        out = cfg.merge_dir / f"groups_{idx}.json"
        json.dump(groups, open(out, "w"))
        rows = {r["i"]: r for r in json.load(open(f))}
        dur_max = max((rows[g[-1]]["end"] - rows[g[0]]["start"] for g in groups), default=0.0)
        print(f"[{idx}] groups={len(groups)} max_dur={dur_max:.1f}s -> {out.name}", flush=True)


if __name__ == "__main__":
    from .config import load_project
    run(load_project(sys.argv[1]), only=sys.argv[2:])
