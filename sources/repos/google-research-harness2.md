# Google Research Harness² 仓库

- **标题：** Harness²: Recursive Agent Harnessing for an Open World
- **类型：** repo
- **组织：** Google Research / Google Cloud AI Research（项目页与论文）
- **代码：** <https://github.com/google-research/harness2>
- **项目页：** <https://harness2.github.io/>
- **论文：** [Harness²（arXiv:2610.14224）](../papers/harness2_arxiv_2610_14224.md)
- **许可：** Apache-2.0；第三方目录及 benchmark / 数据依赖按各自上游许可
- **核验日期：** 2026-10-22
- **一句话说明：** 一个在模型权重冻结时递归改进 agent 外层 harness 的开源框架，含 Propose–Probe–Compose、跨任务经验库、benchmark adapter 与命令行工具。
- **沉淀到 wiki：** 是 → [Harness² 实体页](../../wiki/entities/paper-harness2.md)

## 开源状态核查

**已开源，可运行。** 官方仓库包含 Python 包、CLI、demo、测试和 benchmark 安装/执行脚本。README 提供无需模型凭证的确定性 demo 命令：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[render]'
harness2-demo --output runs/demo --k 2
```

模型驱动 benchmark 需要 Linux、Python 3.12、相应依赖与模型凭证；README 明确提醒模型调用会产生费用。可先用 `--dry-run` 检查任务选择，再小规模运行。

## 实现入口与运行时序

- `harness2/cli/run_sweep.py`：命令入口，检查配置后创建 adapter、Proposer 并调用 sweep。
- `harness2/core/`：调度与改进流程。
- `harness2/benchmarks/`：JobBench、LAB、WorkBuddy-Bench 任务适配。
- `harness2-demo`：无模型凭证的确定性演示。
- `harness2-judge` / `harness2-report`：独立评分与结果汇总。

## 使用边界

官方实验指南提示 benchmark rubric 可能从公开网络访问；真正在 benchmark 上评估前应设置执行器网络出口限制，以降低答案或评分标准泄漏风险。Harness² 本身不限制出网，且 Codex executor 的 sandbox 关闭。该仓库的实验基准是办公、职业任务与法律工作，不是机器人真机基准。

## 官方入口

- README：<https://github.com/google-research/harness2>
- LICENSE：<https://github.com/google-research/harness2/blob/main/LICENSE>
- 实验指南：<https://github.com/google-research/harness2/blob/main/docs/EXPERIMENTS.md>
- 论文：<https://arxiv.org/abs/2610.14224>
- 项目页：<https://harness2.github.io/>
