# eval-of-gpt-6-astra-as-policy（GPT as Policy）

> 来源归档

- **标题：** GPT 6 Astra as an Embodied Policy（仓库名：GPT-as-Policy）
- **类型：** repo / evaluation toolkit
- **代码：** <https://github.com/anonymous-report-421/eval-of-gpt-6-astra-as-policy>
- **主页：** <https://anonymous-report-421.github.io/public-website/?view=1>
- **Stars：** ~160（2026-09-14）
- **License：** MIT
- **作者：** Yu-Mool Shu、Lipxin Zheng
- **入库日期：** 2026-09-14
- **一句话说明：** RoboDojo / RoboLab 上评测 **GPT 6 Astra Direct** 与 **π0.5 + GPT 6 Astra** 混合策略的开源集成：仿真调度、Codex 后端、双语报告构建与 `public_results/` 对齐种子。

## 仓库结构（README）

```text
hybrid_rollout/
  robodojo/               simulator, policy server, controller, skills and scheduler
  robolab/                separate RoboLab integration
  report_site/            bilingual report source, gallery and packaging tools
  assets/                 licensed visualization fonts
report_web/               prebuilt report and its 21 article clips
public_results/           selected case metadata, scores, seeds and provenance
licenses/                 third-party notices
```

## 运行要点（策展）

| 项 | 说明 |
|----|------|
| 模型 | 固定 `gpt-6-astra` + `xhigh`；需自备授权网关 |
| π0.5 | RoboDojo 任务微调 OpenPI/JAX checkpoint；混合模式 `ROLLOUT_EVALUATION_METHOD=pi05_plus_gpt` |
| Direct | `ROLLOUT_EVALUATION_METHOD=gpt_only`（内部 CLI 名，不启动 π0.5） |
| 仿真 | Isaac Sim 5.1 + RoboDojo 上游资产；见 `hybrid_rollout/robodojo/SOURCE.json` |
| 预览 | `python3 -m hybrid_rollout.report_site.preview --directory report_web --port 8768` |
| 安全 | 不含凭证、原始会话日志或 checkpoint；网关 URL 为 `.invalid` 占位符 |

## 对 wiki 的映射

- 实体页：[GPT 6 Astra 具身策略评测](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md)
- 项目页：[gpt-6-astra-embodied-policy-eval.md](../sites/gpt-6-astra-embodied-policy-eval.md)
- 基准：[RoboDojo](../../wiki/entities/robodojo.md)
