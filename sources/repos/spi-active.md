# SPI-Active（LeCAR-Lab/SPI-Active）

> 来源归档（仓库 README / `active_sysid.md` / `downstream_tasks.md` 要点摘录，非全文镜像）

- **标题：** SPI-Active — Sampling-Based System Identification with Active Exploration
- **类型：** repo
- **组织 / 作者：** LeCAR Lab（CMU）— Nikhil Sobanbabu, Guanqi He, Tairan He, Yuxiang Yang, Guanya Shi
- **代码：** <https://github.com/LeCAR-Lab/SPI-Active>
- **论文：** <https://arxiv.org/abs/2505.14266>（CoRL 2025 Oral；PMLR v305）
- **项目页：** <https://lecar-lab.github.io/spi-active_/>
- **视频：** <https://youtu.be/pxyig4D1ZFs>
- **许可：** MIT（`pyproject.toml` / README badge；GitHub license 元数据为空、根目录未见 `LICENSE` 文件）
- **入库日期：** 2026-07-31
- **一句话说明：** CoRL 2025 Oral 官方实现：Isaac Gym 上用 **采样式 SysID（SPI）+ 主动探索（最大化 FIM）** 辨识 Unitree Go2 的 base mass / CoM / 惯量与模块化电机模型，并提供下游 locomotion 训练脚本；结构对齐 HumanoidVerse / ASAP。

## 开源状态（项目页 + 仓库核查，2026-07-31）

| 模块 | 状态（README TODO） |
|------|---------------------|
| SPI 辨识代码（mass landscape / mass opt / 数据采集脚本） | **已发布** |
| Active Exploration（`spigym/run_active_sysid.py` + omni controller 训练） | **已发布** |
| Downstream task training（速度跟踪 / 前跳 / 偏航跳 / 姿态跟踪） | **已发布** |
| Dataset Replay and Visualize | **待发布** |
| Sim2real（真机部署桥） | **待发布** |

- **结论：部分开源。** 仿真侧辨识与下游训练可跑；优化命令后的「真机采数闭环说明」与「Sim2real 部署」仍标 Stay tuned / TODO。
- **项目页**（<https://lecar-lab.github.io/spi-active_/>）展示方法叙事与真机对比视频；**未另挂独立权重/数据集下载**，以 GitHub 仓为主入口。

## 依赖与运行面（README 声明）

- Ubuntu 22.04 LTS（推荐）、NVIDIA GPU + CUDA、**Python 3.8**、[`uv`](https://docs.astral.sh/uv/)
- **Isaac Gym Preview 4**（需自行下载后 `uv pip install -e isaacgym/python`）
- 安装：`uv venv -p 3.8` → `uv sync --dev` → `uv pip install -e isaac_utils/`
- 代码结构声明对齐 [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse) / [ASAP](https://github.com/LeCAR-Lab/ASAP)

## 入口速查（对齐 README / active_sysid.md）

| 路径 / 命令 | 作用 |
|-------------|------|
| `scripts/data/{walk,jump,stand,sine}.py` | 轨迹数据采集（`walk.py` 需挂到 `unitree_rl_gym`） |
| `scripts/mass_landscape.py` | 不同质量假设下的预测误差景观 |
| `scripts/mass_opt.py` | Bayesian / 采样优化求最优 base mass |
| `spigym/train_agent.py +exp=go2_omni` | 训练多行为 omni locomotion（Walk These Ways 风格） |
| `spigym/run_active_sysid.py +exp=active_sysid` | CMA-ES / Optuna 优化指令序列以最大化 FIM → `best_commands.npz` |
| `spigym/train_agent.py +exp=go2_{locomotion,block_jump,rp_track}` | 下游任务训练（见 `spigym/envs/downstream_tasks.md`） |
| `spigym/agents/sysid/active_sysid.py` | Active SysID 算法实现 |
| `spigym/config/env/active_sysid_openloop.yaml` | `default_param` / `exploration_params` / `motor_model` |
| `spigym/config/robot/g1/g1_23dof_sysid.yaml` | G1 人形 SysID 配置线索（泛化实验） |

**示例（README mass opt）：** `config=all`、horizon=5、50 trials → GT base mass 6.921 kg，最优 7.006 kg，best cost ≈ 0.028。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [SPI-Active 实体](../../wiki/entities/paper-notebook-sampling-based-system-identification-with-active.md) | 方法与开源边界归纳 |
| [System Identification](../../wiki/concepts/system-identification.md) | 腿足采样式 + 主动探索 SysID 实例 |
| [CMA-ES](../../wiki/methods/cma-es.md) | SPI / Active 阶段的黑箱优化器 |
| [PACE](../../wiki/entities/paper-pace-sim2real-legged-robots.md) | 同为足式 SysID→RL；PACE 偏悬空关节参数，SPI-Active 偏 base 惯量 + 主动激励 |
| [SAGE](../../wiki/entities/sage-sim2real-actuator-gap-estimator.md) | gap **度量**；SPI-Active 做参数 **反推** |
| [FADA](../../wiki/entities/paper-fada-humanoid.md) | 同 LeCAR / CMU 线：FADA 做执行层少样本适应，SPI-Active 做仿真参数辨识 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/spi_active_arxiv_2505_14266.md`](../papers/spi_active_arxiv_2505_14266.md)
- 项目页：[`sources/sites/spi-active.md`](../sites/spi-active.md)
- Paper Notebooks 锚点：[`sources/papers/humanoid_pnb_spi-active.md`](../papers/humanoid_pnb_spi-active.md)
- 沉淀 **[`wiki/entities/paper-notebook-sampling-based-system-identification-with-active.md`](../../wiki/entities/paper-notebook-sampling-based-system-identification-with-active.md)**
