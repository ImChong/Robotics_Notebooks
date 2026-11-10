# RoboCode — AgenticGenPlan 官方开源实验仓库

> 源码一手归档（repo，2026-11-10）

- **代码：** <https://github.com/tomsilver/robocode>
- **主页：** <https://agenticgentamp.github.io/>
- **论文：** <https://arxiv.org/abs/2609.30233>
- **状态：** 已开源，MIT；包含 Python 源码、Hydra 配置、Docker 严格黑盒构建、基线与分析脚本。
- **核对范围：** 阅读仓库 README 的 Installation、Quickstart、Paper experiments、Running and evaluating、Saved artifacts；读取 `experiments/run_experiment.py`、`src/robocode/approaches/agentic_approach.py`、`agentic_base.py`、`base_approach.py`、`src/robocode/utils/episode.py` 以及 `src/robocode/prompts.py` 的关键运行入口和调用方法。未逐行检查所有模块，也未实际发起付费 Agent/API 复现实验。
- **通读：** 已完成
- **原文：** [论文原文归档](../papers/agenticgenplan_arxiv_2609_30233.md)
- **对应 wiki：** [AgenticGenPlan](../../wiki/entities/paper-agenticgenplan-generalized-tamp.md)

## 复现入口与实际运行时序

1. `experiments/run_experiment.py::_main`（Hydra）解析实验设置、评测种子，构造 env 与 approach，写出 env_description.md。
2. 泛化模式调用 `approach.train()` → `AgenticApproach.train`：拼出 prompt、启动 `_run_sandbox`。
3. `GeneratedProgramApproach._run_sandbox` 在严格黑盒模式借 `env_server_running` 启动不可见的环境服务端，给隔离容器客户端提供 reset/step 等交互接口；调用后端运行编码代理，写出 `sandbox/approach.py`。
4. `AgenticApproach.train` 调用 `_load_generated`（实际是 `load_generated_approach`）加载已生成程序。
5. `run_experiment._main` 在训练后才派生 held-out eval_seeds，逐个调用 `run_episode_with_timeout`；后者的 `run_episode` 先 `env.reset`，再 `approach.reset`，循环 `approach.step` → `GeneratedProgramApproach._get_action` → `generated.get_action` → `env.step` → `approach.update`，到终止/超时为止。
6. 记录 `results.json`、`sandbox/agent_log.txt`；评测期**不运行**合成 agent。

## 官方 Quickstart（README）

```bash
# Python 3.11, git submodule update --init --recursive, uv sync --dev
bash docker/build_strict_blackbox.sh
python experiments/run_experiment.py \
  approach=agentic approach/backend=claude_opus5 \
  environment=motion2d_generalized primitive_level=none \
  approach.container_backend=docker \
  approach.blackbox=true approach.blackbox_strict=true \
  approach.max_budget_usd=20.0 \
  replicate_seed=42 eval_seed="$EVAL_SEED" \
  num_eval_tasks=100 eval_timeout=60
```

注意：`EVAL_SEED` 应由复现者设定且对 coding agent 保密，论文评测 seed 需向作者索取；不应用随机新 seed 冒充论文逐例复现。官方 README 也指出 `approach=agentic` 单独使用时并**不**自动启用严格黑盒配置。

## 可追溯代码入口

- [run_experiment.py](https://github.com/tomsilver/robocode/blob/main/experiments/run_experiment.py)
- [agentic_approach.py](https://github.com/tomsilver/robocode/blob/main/src/robocode/approaches/agentic_approach.py)
- [agentic_base.py](https://github.com/tomsilver/robocode/blob/main/src/robocode/approaches/agentic_base.py)
- [episode.py](https://github.com/tomsilver/robocode/blob/main/src/robocode/utils/episode.py)
- [prompts.py](https://github.com/tomsilver/robocode/blob/main/src/robocode/prompts.py)
- [README](https://github.com/tomsilver/robocode/blob/main/README.md)
