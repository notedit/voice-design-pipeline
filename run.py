#!/usr/bin/env python3
"""统一入口:python run.py <stage> <project> [参数...]

stage(括号内为兼容的旧名):
  extract          分离BGM+VAD+锁定目标说话人+转写(重,GPU),
                   写 work/reports/report_XXX.json(旧名 stage1)
  group            report -> 归组输入 + 贪心归组基线(旧名 merge-prep)
  group-llm        [ep_idx...]  LLM 语义归组复审(可选;旧名 merge-llm)
  group-apply      应用语义拆分 + 校验 -> merge/groups_XXX.json(旧名 merge-apply)
  export           按 groups 重切导出 -> dataset/{wavs,metadata.csv,filelist.txt}
                   (旧名 cut)
  align            全量强制对齐 -> alignments.jsonl
  fix-punct        按实测停顿校正标点(旧名 punct-fix)
  fix-tail-punct   尾标点如实化:被强制断开的条目结尾"。"改","(旧名 tail-punct)
  remove-nonspeech [--apply]  剪除未转写的副语言声音(笑声/语气声等),
                   响度+对齐驱动,无模型依赖;默认只出报告+预览,
                   --apply 就地改写并需重跑 align(旧名 para-cut)
  qa-silence       [dataset_dir] [--apply]  静音异常检测(+ 可选剔除)
                   (旧名 silence-qa)
  denoise          [dataset_dir] [--apply]  语音增强降噪(MossFormer2_SE_48K),
                   带 ECAPA 音色验收,低于阈值保留原音频
  loudnorm         [dataset_dir] [--apply]  音量归一化(项目内集级增益)
  qa-llm           [N|ids.txt]  LLM 打分+判定(旧名 llm-audit)
  qa-prosody       [dataset_dir] [--apply]  声学韵律指标(旧名 prosody)
  all              依次跑 extract -> group -> group-apply -> export -> align
                   -> fix-punct(可选/需人工复核的阶段不自动跑)

用法示例:
  python run.py extract kuaidao
  python run.py extract kuaidao 003 004     # 只处理指定集号
  python run.py export wowan
  python run.py qa-silence kuaidao dataset --apply
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
    STAGES = ["extract", "group", "group-llm", "group-apply", "export",
              "align", "fix-punct", "fix-tail-punct", "remove-nonspeech",
              "qa-silence", "denoise", "loudnorm", "qa-llm", "qa-prosody", "all"]
    LEGACY = {"stage1": "extract", "merge-prep": "group",
              "merge-llm": "group-llm", "merge-apply": "group-apply",
              "cut": "export", "punct-fix": "fix-punct",
              "tail-punct": "fix-tail-punct", "para-cut": "remove-nonspeech",
              "silence-qa": "qa-silence", "llm-audit": "qa-llm",
              "prosody": "qa-prosody"}
    ap.add_argument("stage", choices=STAGES + sorted(LEGACY))
    ap.add_argument("project", help=f"projects/<name>.json,可用: {list_projects()}")
    ap.add_argument("rest", nargs="*", help="各 stage 的额外参数,见上面用法说明")
    ap.add_argument("--apply", action="store_true",
                    help="silence-qa:剔除被标记的条目;loudnorm:就地改写 wav;"
                         "prosody:剔除语速过快的条目(默认只出报告不剔除)")
    args = ap.parse_args()

    cfg = load_project(args.project)
    args.stage = LEGACY.get(args.stage, args.stage)

    if args.stage == "extract":
        from ttspipe import stage1
        stage1.run(cfg, only=args.rest)

    elif args.stage == "group":
        from ttspipe import merge_prep
        merge_prep.run(cfg)

    elif args.stage == "group-llm":
        from ttspipe import merge_llm
        merge_llm.run(cfg, only=args.rest)

    elif args.stage == "group-apply":
        from ttspipe import merge_apply
        merge_apply.run(cfg)

    elif args.stage == "export":
        from ttspipe import cut
        cut.run(cfg)

    elif args.stage == "align":
        from ttspipe import align
        align.run(cfg)

    elif args.stage == "fix-punct":
        from ttspipe import punct_fix
        punct_fix.run(cfg)

    elif args.stage == "fix-tail-punct":
        from ttspipe import tail_punct
        tail_punct.run(cfg)

    elif args.stage == "remove-nonspeech":
        from ttspipe import para_cut
        para_cut.run(cfg, apply=args.apply)

    elif args.stage == "qa-silence":
        from ttspipe import silence_qa
        dataset_dir = (cfg.out_path / args.rest[0]) if args.rest else None
        silence_qa.run(cfg, dataset_dir=dataset_dir, apply=args.apply)

    elif args.stage == "denoise":
        from ttspipe import denoise
        dataset_dir = (cfg.out_path / args.rest[0]) if args.rest else None
        denoise.run(cfg, dataset_dir=dataset_dir, apply=args.apply)

    elif args.stage == "loudnorm":
        from ttspipe import loudnorm
        dataset_dir = (cfg.out_path / args.rest[0]) if args.rest else None
        loudnorm.run(cfg, dataset_dir=dataset_dir, apply=args.apply)

    elif args.stage == "qa-llm":
        from ttspipe import llm_audit
        sel = None
        if args.rest:
            sel = int(args.rest[0]) if args.rest[0].isdigit() else args.rest[0]
        llm_audit.run(cfg, selector=sel)

    elif args.stage == "qa-prosody":
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
