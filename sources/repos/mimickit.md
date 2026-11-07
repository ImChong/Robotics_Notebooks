# Source: MimicKit (xbpeng/MimicKit)

> 来源归档（官方 README / Starter Guide 论文 / 关键源码通读）

- **Title**: MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control
- **URL**: https://github.com/xbpeng/MimicKit
- **Project page**: https://motion.stanford.edu/research/mimickit
- **Paper (Starter Guide)**: https://arxiv.org/abs/2510.13794
- **Author**: Xue Bin Peng（论文署名单位：Simon Fraser University 与 NVIDIA）
- **入库日期**：2026-11-07（本次通读核对；仓库规模等信息为 2026-07 核查）
- **核对范围：** 2026-11-07 读完：官方 `README.md` 全文；Starter Guide 论文 arXiv:2510.13794（v4，2026-01-18）全文（§1–§7、表 1、结论）；`mimickit/run.py` 全文；`envs/base_env.py`、`envs/env_builder.py`、`learning/agent_builder.py`；`learning/base_agent.py` 的函数结构与训练 / 测试入口；`docs/README_AWR.md`、`README_LCP.md`、`README_SMP.md`；`LICENSE` 开头确认 Apache-2.0。**没有读**：各 Agent / Env 子类的具体实现（DeepMimic 奖励项、AMP / ADD 判别器网络、ASE 潜空间等）、引擎适配层源码、`tools/` 重定向脚本；没有安装与运行任何训练，也没有核对 SharePoint 上的数据包内容。
- **通读：** 已完成
- **Year**: 2025
- **Type**: Codebase / Research Framework
- **License**: Apache-2.0
- **规模**：约 2.2k stars（2026-07 核查），最近推送 2026-06-23，主分支 `main`

## 核心内容（仓库 README 归纳）

- **定位**：轻量级、模块化的强化学习框架，专为物理机器人/角色的运动模仿与控制设计；强调「clean and lightweight, minimal dependencies」。更大规模、更全栈的姊妹框架为 [ProtoMotions](https://github.com/NVlabs/ProtoMotions/)。
- **集成算法**：DeepMimic, AMP, AWR, ASE, LCP, ADD, SMP（每个算法在 `docs/README_<算法>.md` 有独立说明）。

### 多仿真后端（Engines）

| 后端 | 用法 | 版本基线（README 标注） |
|------|------|------------------------|
| Isaac Gym | `--engine_config data/engines/isaac_gym_engine.yaml` | 官方 Isaac Gym 发行版 |
| Isaac Lab | `--engine_config data/engines/isaac_lab_engine.yaml` | 测试过 commit `2ed331acfcbb1b96c47b190564476511836c3754` |
| Newton | `--engine_config data/engines/newton_engine.yaml` | 测试过 `v1.0.0` |

官方建议为每个仿真器用 Conda 建独立 Python 环境；安装后 `pip install -r requirements.txt`，再从作者提供的 SharePoint 打包下载 assets 与 motion data 解压到 `data/`。

### 训练 / 测试 / 分布式

- 训练入口：`python mimickit/run.py --mode train --num_envs 4096 --engine_config ... --env_config ... --agent_config ... --visualize true --out_dir output/`。
- 参数可收敛到 `arg_file`（如 `args/deepmimic_humanoid_ppo_args.txt`），与命令行等价；全算法预置参数在 `args/`。
- 测试：`--mode test --model_file <pt>`；**预训练模型**在 `data/models/`，对应训练日志在 `data/logs/`。
- **分布式训练**：`--devices cuda:0 cuda:1 ...` 支持多 CPU / 多 GPU 多进程并行。
- **日志**：`--logger` 支持 `txt` / TensorBoard `tb` / `wandb`；`--video true` 可无头录制视频；`tools/plot_log/plot_log.py` 可绘制 `log.txt` 曲线。
- 可视化交互：`Alt`+左键平移相机、滚轮缩放、`Enter` 暂停、`Space` 单步。

### 动作数据与重定向

- 动作以 `.pkl` 存储（`mimickit/anim/motion.py` 的 `Motion` 类）：每帧为 `[root position (3D), root rotation (3D), joint rotations]`，3D 旋转用 **3D 指数映射**（Exponential Maps），关节顺序为运动学树深度优先遍历。
- `motion_file` 既可指单条 clip，也可指 `data/datasets/` 下的 **数据集文件**（多 clip 混合训练）。
- `view_motion` 环境可回放动作 clip 做数据检查。
- **重定向工具链**：
 - `tools/gmr_to_mimickit/` — 将 [GMR](https://github.com/YanjieZe/GMR) 输出转为 MimicKit 格式；
 - `tools/smpl_to_mimickit/` — 将 [AMASS](https://amass.is.tue.mpg.de/) 的 SMPL 动作转为 MimicKit 格式。

### 策略观测：tan_norm

DeepMimic / AMP 等环境的 `root_rot_obs` 与 `joint_rot_obs` 使用 **tan_norm**（6 维：旋转后的参考切向 + 法向），实现见 `mimickit/util/torch_util.py`；摘录与 Zhou 6D 对照见 [`sources/repos/mimickit_tan_norm.md`](mimickit_tan_norm.md)，wiki 页 [`wiki/formalizations/tan-norm-rotation.md`](../../wiki/formalizations/tan-norm-rotation.md)。

## Starter Guide 论文核对要点（arXiv:2510.13794 v4，2026-11-07）

- **四个组件（§3）：** Agent（学习算法与经验缓冲）、Model（网络结构；actor-critic 可含策略与价值网络，AMP 另含判别器）、Environment（任务逻辑、观测与奖励、`step` 返回 obs/reward/done/info）、Engine（统一的底层仿真 API，目前支持 Isaac Gym、Isaac Lab、Newton；论文称将来会加其他仿真器与真机部署）。
- **done 标志（§3.3，`base_env.py` 的 `DoneFlags`）：** `NULL`（未终止）、`FAIL`（失败，如摔倒，可加终止惩罚）、`SUCC`（成功，可加终止奖励）、`TIME`（超时，按理应继续，可用价值函数 bootstrap 以模拟无限时域 MDP）。
- **Engine 控制模式（§3.4，Isaac Gym 引擎）：** `none`、`pos`（PD 目标旋转，支持 1D 转动与 3D 球关节）、`vel`、`torque`、`pd_1d`（仅限全 1D 转动关节的形态）。
- **方法（§4）：** DeepMimic（通用可靠、精确复现参考动作，但策略被限制在贴近参考；"通常是考虑更复杂技术之前的好起点"）；AMP（对抗式分布匹配，学风格而非逐帧跟踪，更灵活但更易陷入局部最优，难复现高动态动作）；ASE（对抗模仿 + 互信息技能发现，学可复用的潜技能空间）；ADD（对抗式运动跟踪，用差分判别器自动学习自适应的跟踪目标，减轻手工调跟踪奖励的负担）。大多数方法以 PPO 为底层 RL；因 PPO 是 on-policy、样本效率差，框架另提供 off-policy 的 AWR 作为替代。LCP、SMP 在 README 与 `docs/` 中有说明，Starter Guide 正文未单独介绍。
- **实验（§7，表 1、图 5）：** 全部在 Isaac Gym 上，人形角色对多段动捕片段（Run、Jog、Sideflip、Crawl、Roll、Getup Facedown、Spinkick、Cartwheel、Backflip、Dance A、Walk）分别训练；为公平比较，训练中关闭了 Peng 等 2018 的「姿态偏差终止」，仅在非期望触地时早停；位置跟踪误差与 DoF 速度跟踪误差取 5 个种子、每个模型 4096 个测试回合的均值；结论：AMP 跟踪精度最差（但定性上仍能再现整体行为），DeepMimic 与 ADD 精确，ADD 在不同动作间更一致；论文强调**启用姿态偏差终止后**跟踪精度与一致性通常显著提升，因此跟踪类方法默认配置带该终止。还展示了 SMPL 人形、Unitree G1、Unitree Go2 四足的技能（如 G1 的 Run / Cartwheel、Go2 的 Trot / Canter）。
- **README 补充：** 训练日志 `--logger` 取 `txt` / `tb` / `wandb`；`run.py` 默认 `num_envs=1`、`logger=txt`、`visualize=True`；`--devices` 多卡启动多进程，每个进程按 rank 偏移随机种子；`agent_builder` 支持的 agent 名：PPO、AWR、AMP、SMP、ASE、ADD、LCP；`env_builder` 支持 char、deepmimic、amp、ase、add、smp 及 location / steering / dodgeball 任务环境、view_motion 等。
- **LCP 文档：** LCP agent 是一个可继承其他 agent（默认 PPO）的包装类，对 actor 加 Lipschitz 平滑损失，关键超参为 `lcp_weight`。
- **SMP 文档：** 用预训练 LaFAN1 先验训练 location / steering / dodgeball 任务策略，策略训练阶段不使用动作数据，默认开启 Generative State Initialization（`enable_gsi: True`）；新先验用 `tools/diffusion_model/train_tinymdm.py` 训练。

## 与 ProtoMotions 的关系

- **MimicKit**：偏算法与论文复现的轻量框架，运动模仿方法族谱集中、依赖少。
- **ProtoMotions**：偏大规模并行仿真、多后端、数据管线与部署导出的全栈研究平台；README 自述「更 feature-rich」。

## BibTeX（仓库提供）

```bibtex
@article{MimicKitPeng2025,
  title={MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control},
  author={Peng, Xue Bin},
  year={2025},
  eprint={2510.13794},
  archivePrefix={arXiv},
  primaryClass={cs.GR},
  url={https://arxiv.org/abs/2510.13794},
}
```
