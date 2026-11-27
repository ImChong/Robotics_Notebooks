# RHO 核心优化器官方仓库归档（KE7/helix）

- **仓库地址：** <https://github.com/KE7/helix>
- **仓库全称：** `KE7/helix`（PyPI 包名：`helix-evo`）
- **许可证：** BSD-3-Clause
- **默认分支：** `main`
- **关联论文 / 项目：** [RHO: Your Coding Agent is Secretly a Roboticist](https://arxiv.org/abs/2606.16458)（[项目页](https://rho-robotics.github.io)）；仓库 README 的「Built with HELIX」栏直接链接该论文及其项目页，并作为 RHO 论文第 1、3 节指明的反射演化优化器实现。
- **代码核对范围与代表性文件（基于 `context.json` `files_sample` 与 `readme_preview` 静态核对）：**
  - 核心文档与配置：`README.md`、`AGENTS.md`、`CLAUDE.md`、`docs/concepts/how-it-works.md`、`docs/getting-started/quickstart.md`、`docs/getting-started/evaluator.md`、`docs/reference/architecture.md`、`docs/reference/cli.md`、`docs/guides/sandboxing.md`、`docs/guides/ui.md`。
  - 隔离与智能体后端容器：`docker/base.Dockerfile`、`docker/claude.Dockerfile`、`docker/codex.Dockerfile`、`docker/cursor.Dockerfile`、`docker/opencode.Dockerfile`、`docker/pi.Dockerfile`、`docker/agy.Dockerfile`。
  - 机器人操作示例（ManiSkill 3）：`examples/maniskill_pick_place/README.md`、`examples/maniskill_pick_place/API_REFERENCE.md`、`examples/maniskill_pick_place/check.py`；演化结果与评估归档位于 `docs/experiments/maniskill-pick-place/evolved/solver/`（含 `__init__.py`、`main.py`、`common.py`、`tasks/pickcube.py`、`tasks/pushcube.py`、`tasks/stackcube.py`）及 `reference_policy.py`、`test_benchmark.py`、`render_rollouts.py`。
  - 其他基准示例：`examples/circle_packing/`（`helix.toml`、`evaluate.py`、`run_main.py`、`solve.py`）与 `examples/eterna100/`（`helix.toml`、`evaluate.py`、`designer.py`）。
- **可核实运行入口（来自 README Quickstart）：**
  - 安装与初始化：`pip install helix-evo` → `helix init`（生成 `helix.toml` 配置文件，定义优化目标与评测命令）。
  - 演化与导出：`helix evolve`（在 `.helix/worktrees/` 独立 git worktree 中调度编程智能体进行多轮编辑并调用 `evaluate.py`）→ `helix frontier` / `helix ui` → `helix best --export ./best`。
  - 评测脚本契约：`evaluate.py` 执行候选仓库并向标准输出打印 `HELIX_RESULT=` 前缀的单行 JSON 结果列表（含标量得分与 `{"feedback": ...}` 诊断信息）。
- **开源覆盖边界与复现状态：**
  - **边界说明：** `KE7/helix` 完整开源了 RHO 所依赖的通用仓库级反射演化优化器（HELIX / `helix-evo`）及 ManiSkill 3 操作示例；而论文实验中的 CaP-Bench（Robosuite / LIBERO-PRO）REaaS sidecar 容器代理、RAI O3DE 评测适配代码以及 `judge_prompt.md` 等未在当前静态核对的文件样本中完整出现。
  - **复现状态：** 静态代码核对，未本地运行。

- 对 wiki 的映射：[RHO: Your Coding Agent is Secretly a Roboticist](../../wiki/entities/paper-agentic-2606-16458.md)
