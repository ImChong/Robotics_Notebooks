# RoboCode（AgenticGenPlan）官方代码与运行入口审读

- **仓库：** https://github.com/tomsilver/robocode
- **项目网站：** https://agenticgentamp.github.io/
- **论文：** https://arxiv.org/abs/2609.30233
- **数据：** https://huggingface.co/datasets/prpl-group/agentic-gentamp-runs
- **读取快照：** 上游 `tomsilver/robocode@d3d42dd12e74c00383ceeee5e75f16bd8c4abc09`（main，2026-11-10）
- **许可证：** MIT（上游 LICENSE）
- **开源状态：** 已开源；含代码代理合成 harness、KinDER/PDDLStream 配置、环境黑盒客户端与服务端、评估器、运行记录工具和作者公开数据。官方严格黑盒选项需显式设置。
- **核对范围：** 阅读 README 的 Installation、Quickstart、Paper experiments/Environment map/Methods & settings、评估 seed 与 strict-black-box 注意事项；查看源码 `experiments/run_experiment.py` 的 `_main`、`src/robocode/approaches/agentic_approach.py` 的 `train`、`src/robocode/approaches/agentic_base.py` 的 `_run_sandbox`/`_load_generated`/`reset`/`_get_action`、`src/robocode/utils/env_client.py` 的 `BlackboxEnv`/`make_env`、`src/robocode/utils/episode.py` 的 `load_generated_approach`/`run_episode_with_timeout`，并核查 `src/robocode/utils/backends/claude.py` 和 `codex.py` 的代理 CLI 入口。这里记录已核实的运行调用链，未独立安装/执行完整合成任务。
- **通读：** 已完成

## 复现入口：严格黑盒 vs 默认配置

README 要求 Python 3.11，Git 子模块、`uv sync --dev` 和可用 Docker；按对应的 Claude Code/Codex CLI 做身份验证，然后运行 `bash docker/build_strict_blackbox.sh` 构建严格黑盒镜像。安装还可能需要 EGL/OpenGL、C/C++ 编译工具与可选 planner extras。**未在本执行环境实测安装/训练。**

```bash
git clone https://github.com/tomsilver/robocode.git
cd robocode
git submodule update --init --recursive
uv sync --dev
source .venv/bin/activate
bash docker/build_strict_blackbox.sh

# eval seed 由实验者自己选择；论文作者的保留 seed 未公开
export EVAL_SEED=123456
python experiments/run_experiment.py \
  approach=agentic approach/backend=claude_opus5 \
  environment=motion2d_generalized primitive_level=none \
  approach.container_backend=docker \
  approach.blackbox=true approach.blackbox_strict=true \
  approach.max_budget_usd=20.0 \
  replicate_seed=42 eval_seed="$EVAL_SEED" \
  num_eval_tasks=100 eval_timeout=60 \
  hydra.run.dir=outputs/agenticgenplan-motion
```

**配置陷阱：** `approach=agentic` 本身**不等于**论文黑盒实验：上游 `experiments/conf/approach/agentic.yaml` 的 `blackbox: false`、`blackbox_strict: false` 为默认；上游默认还可能选 `approach=random`、`environment=small_maze`、`primitive_level=low_level`。严格黑盒需要 `primitive_level=none` + 两项黑盒开关；否则会读源码或依赖预制 primitive。

`EVAL_SEED=123456` 仅是**独立示例种子**，不代表论文结果复现用的隐藏 seed。

## 源码时序（已核实的接口）

1. `experiments/run_experiment.py::_main` 用 Hydra 配置创建 `env`、准备任务说明与 primitives，然后构造 `AgenticApproach`。
2. `AgenticApproach.train` 从 `_build_agentic_prompts` 得到描述/初始化文件，经 `_run_sandbox` 在 Docker/Apptainer 沙箱运行编码代理；黑盒模式下 `env_server_running` 在沙箱外提供模拟器服务。
3. 代理通过沙箱里的 `env_client.make_env` 创建 `BlackboxEnv`，调用 `reset`、`step`、`render_state`/`render_policy` 自主做探针和测试；这些是**合成期** API，不是冻结策略的依赖。
4. 编码代理生成 `sandbox/approach.py`（`GeneratedApproach` 带 `reset` 与 `get_action`）；`AgenticApproach.train` 调用 `_load_generated`，内部由 `episode.load_generated_approach` 装载。
5. 返回 `run_experiment.py` 后，`approach.train()` 只调用一次；其后才生成测试用 seed，再由 `episode.run_episode_with_timeout` 在新初始态循环 `approach.reset`、`approach.get_action` 与 `env.step`。此时不再调用编码代理。
6. 统计每实例成败、计算时间、随物体数量扩展的表现和确定性重放，最终写 `results.json`。

## 安全与评测边界

- 作者使用隔离容器并做红队验证，但自己的复现部署仍需对 API 密钥、Docker 权限、资源上限负责。
- 论文指标来自具备固定物理/接口定义的仿真环境。要迁移到新引擎/机器人，需重新对齐环境状态、动作、接触/碰撞物理与目标判据，再验证；不能拿原策略直接上机。
- `run_experiment.py` 从合成进程隔离评测 seed，测试在 `train()` 返回后采样。独立 seed 的再评估与论文作者 held-out seed 不相同。
- 研究仓库包含多个实验条件、部分更近期的 CLI/backends；README 中复现参数比按模型名推断更可靠。

## 对应知识库

- [arXiv v1 论文归档](../papers/agenticgenplan_coding_agents_generalized_tamp_arxiv_2609_30233.md)
- [官方项目网站归档](../sites/agenticgentamp.md)
- [AgenticGenPlan 结构化实体](../../wiki/entities/paper-agenticgenplan-generalized-tamp-coding-agents.md)
