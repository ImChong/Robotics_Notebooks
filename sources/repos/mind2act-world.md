# Mind2Act_World：诊断评测源码及运行入口

- **仓库：** <https://github.com/Mind2Act/Mind2Act_World>
- **论文：** <https://arxiv.org/abs/2611.26444>
- **官方项目页：** <https://mind2act-world.com/>
- **许可：** 项目原创代码 MIT；依赖与资产按 THIRD_PARTY.md 另行核对
- **入库日期：** 2026-11-22
- **核对范围：** README.md、docs/source-layout.md、docs/agent-runtime.md、docs/submission.md、policies/README.md、eval/cli.py、eval/batch.py、eval/runner/__init__.py、eval/scoring.py、setup.py。
- **通读：** 已完成

## 源码入口与模块（依据官方源码）

| 环节 | 官方位置 | 已核对的职责 |
|---|---|---|
| CLI | `eval/cli.py` → `eval/public_cli.py` | `python -m mind2act_eval.cli run` 选择 `--e2e` 或 `--agent` |
| 配置/实例 | `eval/internal.py` | 校验 instance spec，进入批次 worker |
| 运行监督 | `eval/batch.py` | 独立进程 supervisor、预算、超时、尝试记录 |
| 任务加载 | `task/catalog.py` | 选择 30 个 task ID 的任务逻辑 |
| 仿真 | `env/sim/`、`env/robot/`、`env/variations/` | Isaac Sim 场景、X5 机器人和随机化 |
| 策略接口 | `policies/` | XPolicyLab 端到端策略、`m2a_agent` 工具 Agent |
| rollout | `eval/runner/` | 每步观测→动作→仿真推进，episode 完成时调用 scorer |
| 判分 | `eval/scoring.py` + `task/<id>/` scorer | 物理规则与任务特定 success、部分进度 |
| 导出 | `eval/runner/` + episode storage | 保存结果与评测记录 |

## 官方运行指引（需额外资产和服务）

前置 Linux + NVIDIA GPU + Isaac Sim 5.1 提供的 Python 3.11。下载与当前代码相匹配的 [开发资产版本](https://huggingface.co/datasets/Mind2Act-World/mind2act-dev-assets/tree/main)，根据 `release-v1.0.0/RELEASE.json` 中 `official_code_commit` 对齐源码，并运行 `sha256sum -c SHA256SUMS` 后解包、安装配套 `mind2act_dev` wheel。模型及推理服务另备。

- 端到端：`"$ISAAC_PYTHON" -m mind2act_eval.cli run --spec "$SPEC" --asset-root "$ASSET_ROOT" --e2e --gpu "$GPU"`
- 工具 Agent：相同命令改用 `--agent`，配 `config/agent.json`，调用 `move_eef`、`wait_steps`、`memory_resolve`、`give_up`；其 planner 和模型服务须另配。
- **验证状态：** 仅核实仓库源码与文档入口，未在本次环境安装 Isaac Sim / 资产或执行完整评测。公开源码不等于已经完成可复现 benchmark 运行。

## 知识映射

- [论文通读](../papers/mind2act-world-arxiv-2611-26444.md)
- [官网开放边界](../sites/mind2act-world.md)
- [论文详情页（含源码时序图）](../../wiki/entities/paper-mind2act-world.md)