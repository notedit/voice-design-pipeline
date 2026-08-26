#!/usr/bin/env python3
"""统一入口:python run.py <stage> <project> [参数...]

stage:
  stage1       分离+VAD+说话人聚类+ASR(重),写 work/reports/report_XXX.json
  merge-prep   report -> merge/input_XXX.json + 贪心归组基线 greedy/review
  merge-llm    [ep_idx...]  调 API 做语义归组(可选,不跑也能用贪心基线)
  merge-apply  应用语义拆分(如果有)+ 校验 -> merge/groups_XXX.json
  cut          按 groups 重切音频 -> dataset/{wavs,metadata.csv,filelist.txt}
  align        全量强制对齐 -> alignments.jsonl
  punct-fix    按实测停顿校正标点
  silence-qa   [dataset_dir] [--apply]  静音异常检测(+ 可选剔除)
  loudnorm     [dataset_dir] [--apply]  音量归一化报告(+ 可选就地改写 wav):
               项目内集级增益,拉平集与集之间的响度差,保留集内自然强弱
  llm-audit    [N|ids.txt]  inkling-small 打分+判定(需 OPENROUTER_API_KEY)
  prosody      [dataset_dir] [--apply]  声学韵律指标(+ 可选剔除语速过快条目)
  all          依次跑 stage1 -> merge-prep -> merge-apply -> cut -> align -> punct-fix
               (跳过 merge-llm/llm-audit/prosody/silence-qa——这几步要么要花钱调
               API,要么判定结果需要人工过一遍,不适合无人值守地自动跑完)

用法示例:
  python run.py stage1 kuaidao
  python run.py stage1 kuaidao 003 004      # 只处理指定集号
  python run.py cut wowan
  python run.py silence-qa kuaidao dataset --apply
  python run.py all wowan
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ttspipe.config import list_projects, load_project  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=[
        "stage1", "merge-prep", "merge-llm", "merge-apply", "cut", "align",
        "punct-fix", "silence-qa", "loudnorm", "llm-audit", "prosody", "all"])
    ap.add_argument("project", help=f"projects/<name>.json,可用: {list_projects()}")
    ap.add_argument("rest", nargs="*", help="各 stage 的额外参数,见上面用法说明")
    ap.add_argument("--apply", action="store_true",
                    help="silence-qa:剔除被标记的条目;loudnorm:就地改写 wav;"
                         "prosody:剔除语速过快的条目(默认只出报告不剔除)")
    args = ap.parse_args()

    cfg = load_project(args.project)

    if args.stage == "stage1":
        from ttspipe import stage1
        stage1.run(cfg, only=args.rest)

    elif args.stage == "merge-prep":
        from ttspipe import merge_prep
        merge_prep.run(cfg)

    elif args.stage == "merge-llm":
        from ttspipe import merge_llm
        merge_llm.run(cfg, only=args.rest)

    elif args.stage == "merge-apply":
        from ttspipe import merge_apply
        merge_apply.run(cfg)

    elif args.stage == "cut":
        from ttspipe import cut
        cut.run(cfg)

    elif args.stage == "align":
        from ttspipe import align
        align.run(cfg)

    elif args.stage == "punct-fix":
        from ttspipe import punct_fix
        punct_fix.run(cfg)

    elif args.stage == "silence-qa":
        from ttspipe import silence_qa
        dataset_dir = (cfg.out_path / args.rest[0]) if args.rest else None
        silence_qa.run(cfg, dataset_dir=dataset_dir, apply=args.apply)

    elif args.stage == "loudnorm":
        from ttspipe import loudnorm
        dataset_dir = (cfg.out_path / args.rest[0]) if args.rest else None
        loudnorm.run(cfg, dataset_dir=dataset_dir, apply=args.apply)

    elif args.stage == "llm-audit":
        from ttspipe import llm_audit
        sel = None
        if args.rest:
            sel = int(args.rest[0]) if args.rest[0].isdigit() else args.rest[0]
        llm_audit.run(cfg, selector=sel)

    elif args.stage == "prosody":
        from ttspipe import prosody
        dataset_dir = (cfg.out_path / args.rest[0]) if args.rest else None
        prosody.run(cfg, dataset_dir=dataset_dir, apply=args.apply)

    elif args.stage == "all":
        from ttspipe import stage1, merge_prep, merge_apply, cut, align, punct_fix
        print("== stage1: separate + vad + speaker + asr ==")
        stage1.run(cfg)
        print("== merge-prep: merge inputs + greedy baseline ==")
        merge_prep.run(cfg)
        print("== merge-apply: apply splits (if any) + validate ==")
        merge_apply.run(cfg)
        print("== cut: final dataset ==")
        cut.run(cfg)
        print("== align: forced alignment ==")
        align.run(cfg)
        print("== punct-fix ==")
        punct_fix.run(cfg)
        print("ALL STAGES DONE")


if __name__ == "__main__":
    main()
