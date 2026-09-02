"""纯函数单测:不碰模型、不碰 GPU,秒级跑完。覆盖最容易被"顺手改一下"弄坏的
判定逻辑。"""
import numpy as np
import pytest

from ttspipe.config import CutConfig, MergeReviewConfig, ParaCutConfig
from ttspipe.cut import apply_edge_fades, join_texts
from ttspipe.merge_prep import greedy_groups
from ttspipe.para_cut import apply_edits, plan_utt, shift_items, verify
from ttspipe.stage1 import select_target_cluster
from ttspipe.tail_punct import pair_groups
from ttspipe.textalign import match_chars


def unit(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


# ---------- extract: 选簇 ----------
def test_select_cluster_no_ref_picks_longest():
    embs = np.stack([unit([1, 0, .05]), unit([1, .02, 0]), unit([0, 1, .05]), unit([.02, 1, 0])])
    main, stats = select_target_cluster(embs, [0, 0, 1, 1], [(0, 10), (12, 25), (26, 28), (30, 33)])
    assert main == 0 and stats[0]["dur"] == 23.0 and "ref_dist" not in stats[1]


def test_select_cluster_ref_picks_nearest_even_if_short():
    embs = np.stack([unit([1, 0, .05]), unit([1, .02, 0]), unit([0, 1, .05]), unit([.02, 1, 0])])
    segs = [(0, 10), (12, 25), (26, 28), (30, 33)]
    main, stats = select_target_cluster(embs, [0, 0, 1, 1], segs, ref_emb=unit([0, 1, .02]))
    assert main == 1 and stats[1]["ref_dist"] < 0.05 < stats[0]["ref_dist"]


def test_select_cluster_ref_absent_returns_none():
    embs = np.stack([unit([1, 0, 0]), unit([0, 1, 0])])
    main, _ = select_target_cluster(embs, [0, 1], [(0, 5), (6, 9)], ref_emb=unit([0, 0, 1]))
    assert main is None


# ---------- export: 文本拼接 + 淡入淡出 ----------
def test_join_texts_terminal_fullwidth_no_double_comma():
    assert join_texts(["你好。", "再见"]) == "你好。再见"
    assert join_texts(["你好", "再见"]) == "你好,再见"
    assert join_texts(["你好，", "再见"]) == "你好，再见"   # 全角逗号也算已有标点


def test_edge_fades():
    sr = 16000
    x = np.ones(sr, dtype=np.float32)
    y = apply_edge_fades(x, sr, 0.06, 0.06)
    n = int(0.06 * sr)
    assert y[0] == 0 and abs(y[n] - 1) < 1e-6 and y[-1] < 1e-4
    assert np.all(np.diff(y[:n]) >= 0) and np.all(np.diff(y[-n:]) <= 0)
    assert np.array_equal(apply_edge_fades(x, sr, 0, 0), x)
    assert len(apply_edge_fades(np.ones(10, np.float32), sr, .06, .06)) == 10


# ---------- group: 贪心归组的产品策略 ----------
def rows_from(durs_gaps):
    """[(dur, gap_to_next), ...] -> merge input rows"""
    rows, t = [], 0.0
    for i, (d, g) in enumerate(durs_gaps):
        rows.append(dict(i=i, start=t, end=t + d, dur=d,
                         gap_to_next=g, next_is_adjacent=g is not None, text=f"s{i}"))
        t += d + (g or 0)
    return rows


def test_greedy_old_behavior_packs_to_max():
    mr = MergeReviewConfig(max_gap=2.5, max_dur=30.0, pref_max_dur=None)
    rows = rows_from([(10, .2), (10, .2), (8, .2), (10, None)])
    assert greedy_groups(rows, mr) == [[0, 1, 2], [3]]   # 28.4s 打包,再加就超


def test_greedy_pref_breaks_at_good_gap_in_elastic_zone():
    mr = MergeReviewConfig(max_gap=2.5, max_dur=29.0, pref_max_dur=20.0, pref_min_dur=5.0, good_gap=0.5)
    # 18s 后遇到 0.2s 逗号顿(不好),吞下去到 24s 遇到 0.6s 句末顿 -> 在此断
    rows = rows_from([(18, .2), (6, .6), (5, .2), (5, None)])
    assert greedy_groups(rows, mr) == [[0, 1], [2, 3]]


def test_greedy_pref_backtracks_to_largest_gap_at_hard_cap():
    mr = MergeReviewConfig(max_gap=2.5, max_dur=29.0, pref_max_dur=20.0, pref_min_dur=5.0, good_gap=0.5)
    # 全是短顿,到 29s 硬上限没好断点:回溯到组内最大间隙(0.4s,在 seg1 后)
    rows = rows_from([(10, .1), (8, .4), (8, .1), (8, .1), (5, None)])
    groups = greedy_groups(rows, mr)
    assert groups[0] == [0, 1]
    assert all(rows[g[-1]]["end"] - rows[g[0]]["start"] <= 29.0 for g in groups)


def test_greedy_non_adjacent_always_breaks():
    mr = MergeReviewConfig(pref_max_dur=20.0)
    rows = rows_from([(3, 5.0), (3, None)])   # 间隙 5s > max_gap
    assert greedy_groups(rows, mr) == [[0], [1]]


# ---------- fix-tail-punct: 归组 <-> 条目对应 ----------
def test_pair_groups_skips_dropped_and_tolerates_asr_drift():
    by_i = {0: dict(text="今天天气很好", gap_to_next=0.3, next_is_adjacent=True),
            1: dict(text="短", gap_to_next=0.3, next_is_adjacent=True),      # 会被 cut 丢弃
            2: dict(text="我们去公园散步", gap_to_next=1.2, next_is_adjacent=True),
            3: dict(text="然后回家", gap_to_next=None, next_is_adjacent=False)}
    groups = [[0], [1], [2], [3]]
    ep_texts = ["今天天气很好。", "我们去公园散步，", "然后回家了。"]   # 第三条多了个"了"(重转写差异)
    assert pair_groups(ep_texts, groups, by_i) == [True, False, False]


def test_pair_groups_raises_on_mismatch():
    by_i = {0: dict(text="完全不同的内容", gap_to_next=None, next_is_adjacent=False)}
    with pytest.raises(ValueError):
        pair_groups(["风马牛不相及"], [[0]], by_i)


# ---------- fix-punct: difflib 对齐可重新同步 ----------
def test_match_chars_multichar_token_and_punct():
    items = [["你", 0, .2], ["好", .2, .4], ["SUV", .5, .9], ["车", .9, 1.1]]
    seq, cov = match_chars("你好,SUV车。", items)
    assert cov == 1.0
    assert seq[2][1] is None and seq[-1][1] is None          # 标点不对齐
    assert seq[3][1] is items[2] and seq[5][1] is items[2]   # 多字符 token 的每个字都指向它


def test_match_chars_resyncs_after_mismatch():
    # 旧指针 walk 在"A I"vs"AI"处卡死,后面全丢;difflib 之后能接着对上
    items = [["用", 0, .1], ["AI", .1, .4], ["帮", .4, .5], ["我", .5, .6], ["写", .6, .7]]
    seq, cov = match_chars("用A I帮我写", items)
    assert cov >= 0.8 and seq[-1][1] is items[-1]


# ---------- remove-nonspeech: 刀口规划 ----------
def synth(sr=16000):
    rng = np.random.default_rng(0)
    x = np.zeros(4 * sr, dtype=np.float32)
    def burst(a, b, amp): x[int(a*sr):int(b*sr)] = rng.standard_normal(int((b-a)*sr)).astype(np.float32) * amp
    burst(0.2, 1.2, .3)   # 语音
    burst(1.6, 2.2, .2)   # gap 里的笑声(两侧有静音谷)
    burst(2.6, 3.6, .3)   # 语音
    return x


def test_plan_utt_cuts_gap_event_and_verify_passes():
    sr, x = 16000, synth()
    items = [["a", .2, .7], ["b", .7, 1.2], ["c", 2.6, 3.1], ["d", 3.1, 3.6]]
    cc, pc = CutConfig(energy_floor_db=-42, min_tail_sil=0.3), ParaCutConfig()
    edits, skipped = plan_utt(x, sr, items, cc, pc)
    assert len(edits) == 1 and edits[0]["kind"] == "mid" and not skipped
    assert 1.5 <= edits[0]["t0"] <= 1.65 and 2.15 <= edits[0]["t1"] <= 2.3
    y = apply_edits(x, sr, edits, cc.xfade)
    assert len(y) < len(x)
    ok, _ = verify(y, sr, shift_items(items, edits), cc, pc)
    assert ok


def test_plan_utt_skips_event_glued_to_speech():
    sr = 16000
    x = synth()
    x[int(1.2*sr):int(1.6*sr)] = 0.2   # 笑声直接贴在语音后面,没有静音谷
    items = [["a", .2, .7], ["b", .7, 1.2], ["c", 2.6, 3.1], ["d", 3.1, 3.6]]
    edits, skipped = plan_utt(x, sr, items, CutConfig(min_tail_sil=0.3), ParaCutConfig())
    assert not edits and skipped and skipped[0]["reason"].startswith("no_valley")
