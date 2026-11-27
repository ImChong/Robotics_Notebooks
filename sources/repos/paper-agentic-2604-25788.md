# KinDER 官方开源仓库核对（kindergarden / kinder-baselines）

- **主环境仓库：** [Princeton-Robot-Planning-and-Learning/kindergarden](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden)（默认分支 `main`，MIT License，核对时 56 Stars）
- **配套基线仓库：** [Princeton-Robot-Planning-and-Learning/kinder-baselines](https://github.com/Princeton-Robot-Planning-and-Learning/kinder-baselines)（默认分支 `main`，核对时 14 Stars）
- **历史迁移说明：** 原工作组综合仓库 [prpl-mono](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-mono) README 明确标注 KinDER 环境已迁移至 `kindergarden`，基线已迁移至 `kinder-baselines`；部分底层公共包（如 `bilevel-planning`、`prpl-kinematics`、`relational-structs`）仍在 `prpl-mono` 维护。
- **模型与数据集：** Hugging Face 模型库 `https://huggingface.co/kinder-bench` 与数据集 `https://huggingface.co/datasets/kinder-bench/kinder-datasets`（引用自 `kindergarden/README.md`）。
- **核对边界：** 静态代码核对，未本地运行（基于 GitHub 仓库元数据、目录文件列表采样 `files_sample` 与 `README.md` 预览文本核对，未本地克隆安装或运行测试）。

## 仓库结构与已核实入口

### 1. `kindergarden`（KinDERGarden 与 KinDERGym 核心包）

- **打包与 CI：** `MANIFEST.in`、`LICENSE`（MIT）、`.pylintrc`、`.github/workflows/ci.yml`、`.github/actions/setup/action.yml`、`README.md`。README 说明通过 `./run_ci_checks.sh` 运行完整 CI 检查，支持 Python `>=3.10, <3.13`。
- **分后端安装：** 支持全量 `pip install kindergarden` 或按后端拆分安装（`pip install --no-deps kindergarden` 后安装 `requirements/` 下对应依赖：`kinematic2d.txt`、`dynamic2d.txt`、`kinematic3d.txt`、`kinematic3d_v2.txt`、`dynamic3d.txt`）。其中 `kinematic3d_v2`（`VegaMotion3D`、`VegaPickPlace3D`，基于 Dexmate Vega）依赖 `kindergarden[prpl-kinematics]`。
- **Python API 与目录模块（源自 `README.md` 示例代码与贡献指南）：**
  - 环境注册与构造：`src/kinder/__init__.py` 提供 `kinder.register_all_environments()` 与 `kinder.make("kinder/Obstruction2D-o3-v0")` 等标准 Gymnasium 接口。
  - 对象中心环境基类：`src/kinder/env` 及具体环境模块（如 `kinder.envs.kinematic2d.obstruction2d.ObjectCentricObstruction2DEnv`），支持 `env.observation_space.devectorize(vec_obs)` / `vectorize(...)` 在向量与 `ObjectCentricState` 间互转，并提供 `kinder.NoisyObservation` 与 `kinder.NoisyAction` 包装器。
  - 演示数据与视频脚本：`demos/` 目录存储预采集演示（`files_sample` 已核实 `demos/BaseMotion3D/<seed>/<timestamp>.p` 等 pickle 轨迹文件），`README.md` 注明采集与视频生成脚本位于 `scripts/collect_demos.py` 与 `scripts/generate_demo_video.py`。

### 2. `kinder-baselines`（KinDERBench 基线套件）

- **子包划分（源自 `kinder-baselines/README.md`）：**
  - `kinder-bilevel-planning`（双层规划 TAMP）
  - `kinder-ds-policies`（领域特定手工策略）
  - `kinder-imitation-learning`（模仿学习，如 DP/DPES）
  - `kinder-models`（技能、算子与采样器模型库）
  - `kinder-rl`（PPO / SAC 强化学习）
  - `kinder-vlm-planning`（LLM / VLM 规划）
  - `kinder-trajopt`（MPC 轨迹优化）
  - `kinder-mbrl`（基于学习转移模型的 MBRL）
  - `kinder-pddlstream-planning`（PDDLStream TAMP 基线）
- **已核实入口与实现文件（源自 `files_sample`）：**
  - 主实验入口：`kinder-bilevel-planning/experiments/run_experiment.py`、`kinder-bilevel-planning/experiments/run_noisy_experiments.py`、`kinder-bilevel-planning/experiments/conf/config.yaml` 及 `experiments/conf/env/*.yaml`。
  - 演示采集与结果分析：`kinder-bilevel-planning/experiments/collect_demos_bilevel.py`、`collect_demos_with_skills.py`、`analyze_results.py`、`create_results_dataframe.py`、`gen_multi_run_stats.py`。
  - 规划智能体与环境建模：`kinder-bilevel-planning/src/kinder_bilevel_planning/agent.py`、`injection.py`、`logging_sampler.py`，以及 `env_models/{kinematic2d,dynamic2d,kinematic3d,kinematic3d_v2,dynamic3d}/*.py`。
  - 启动与检查脚本：`kinder-bilevel-planning/scripts/run_*.sh`、`kinder-bilevel-planning/run_ci_checks.sh`。

对 wiki 的映射：[KinDER 实体页](../../wiki/entities/paper-agentic-2604-25788.md)
