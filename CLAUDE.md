# CLAUDE.md

@AGENTS.md

以上引入的 AGENTS.md 是本仓库面向所有 AI agent 的唯一指南(运行方式、数据流、
改代码约束、parity 验证、保密约定)。新增指导原则写进 AGENTS.md,不要在本文件
里另写一份导致两处漂移;本文件只放 Claude Code 特有的补充。

## Claude Code 特有补充

- 跑 `stage1`/`align` 等长任务时用后台方式执行(`run_in_background`),
  期间可继续其他工作;结束后把日志尾部关键行(条数、报告路径)汇报给用户。
- 带 `--apply` 的命令(silence-qa/loudnorm/prosody)和重跑 `cut` 属于
  破坏性操作,执行前必须得到用户明确确认,不适用"自主执行"默认。
