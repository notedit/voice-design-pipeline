#!/usr/bin/env python3
"""文本 <-> 字级时间戳的对齐(difflib,可重新同步)。

align 阶段给出的 items 是对齐器切出的 token([text, start, end]),token
可能是多字符(英文单词、数字串);转写文本 text 里还夹着标点。旧的逐字符
指针 walk 一处对不上就整条错位;这里把 items 展开成字符序列,与 text 做
SequenceMatcher,equal 块内逐字符建立映射,其余字符(标点、规范化差异)
映射为 None——对不上的地方之后能重新同步。

返回 (seq, coverage):seq = [(char, item|None), ...] 覆盖 text 全部字符;
coverage = 被映射到的 item 占比(多字符 item 任一字符映射到即算)。
"""
import difflib


def match_chars(text: str, items):
    expanded, owner = [], []
    for k, it in enumerate(items):
        for ch in str(it[0]):
            expanded.append(ch)
            owner.append(k)
    sm = difflib.SequenceMatcher(None, list(text), expanded, autojunk=False)
    mapping = [None] * len(text)
    hit = set()
    for tag, i0, i1, j0, j1 in sm.get_opcodes():
        if tag != "equal":
            continue
        for di in range(i1 - i0):
            k = owner[j0 + di]
            mapping[i0 + di] = items[k]
            hit.add(k)
    seq = list(zip(text, mapping))
    coverage = len(hit) / len(items) if items else 0.0
    return seq, coverage
