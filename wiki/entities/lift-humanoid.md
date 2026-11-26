---
type: entity
tags:
  - humanoid
  - reinforcement-learning
  - soft-actor-critic
  - model-based-rl
  - sim2real
  - jax
  - brax
  - mujoco-playground
  - booster-robotics
  - unitree
status: complete
updated: 2026-09-15
original:
  - ../../sources/papers/lift_humanoid_arxiv_2601_21363.md
related:
  - ../overview/wam-motion-control-five-paths.md
  - ../tasks/locomotion.md
  - ../methods/reinforcement-learning.md
  - ../methods/model-based-rl.md
  - ../concepts/sim2real.md
  - ../queries/rl-algorithm-selection.md
  - ./brax.md
  - ./mujoco-mjx.md
  - ./mjlab-playground.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_wam_motion_control_five_paths.md
  - ../../sources/papers/lift_humanoid_arxiv_2601_21363.md
  - ../../sources/sites/lift-humanoid-github-io.md
  - ../../sources/repos/bigai-lift-humanoid.md
summary: "LIFT（BIGAI）把人形控制的「墙钟大规模仿真预训练」与「样本高效、较安全的适应微调」串成三阶段：JAX 大规模并行 SAC → 离线训练物理知情（拉格朗日 + 残差）世界模型 → 真机/目标环境确定性采集 + 仅在模型 rollout 中保留随机探索；配套开源与项目页给出 MuJoCo Playground / Brax / Booster T1 等实证叙事。"
---

# LIFT（人形大规模预训练 + 高效微调）

> **名称消歧：** 本页为 BIGAI **人形 RL** LIFT（arXiv:2601.21363）。VLA **力感知后训练** LIFT 见 [paper-lift-reactive-force-vla-posttrain](./paper-lift-reactive-force-vla-posttrain.md)（arXiv:2607.14236）。

**LIFT**（论文缩写：**L**arge-scale pretra**I**ning and efficient **F**ine**T**uning）是面向 **人形机器人 locomotion** 的一条 **强化学习 + 世界模型** 工程管线：先用 **GPU 大规模并行仿真** 把 **离策略 SAC** 训到可部署强度，再用 **物理结构先验 + 学习残差** 的动力学模型支撑 **微调阶段的安全探索布局**。

## 一句话定义

**仿真里用高吞吐 SAC 把策略「推满」，再用物理知情的可微世界模型承接适应——真机只跑确定性动作，随机探索留在模型里。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| JAX | JAX | 支持自动微分与 XLA 编译的数值计算库 |
| SAC | Soft Actor-Critic | 连续控制常用的 off-policy 最大熵算法 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| GPU | Graphics Processing Unit | 图形处理器，大规模并行仿真训练的算力基础 |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| PD | Proportional–Derivative | 关节位置/阻抗底层控制，策略输出常为其 setpoint |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Dreamer | Dreamer (World Models) | 在潜空间想象中训练的 MBRL 代表 |
| DreamerV3 | Dreamer version 3 | 在潜空间想象中训练、单一超参跨 150+ 任务通用的世界模型智能体 |

## 为什么重要

- **对齐两条常见痛点的叙事：** On-policy（如 PPO）在 **万级并行** 下墙钟极快，但 **适应新环境** 时每步数据贵、随机探索对人形 **代价高**；纯 model-based 往往 **墙钟预训练慢** 或 **sim2real 证据链不完整**。LIFT 试图用 **算法 backbone 统一（SAC）+ 阶段解耦** 把两侧各取一段优势。
- **安全叙事具体化：** 微调时 **环境交互策略取均值（确定性）**，**SAC 随机 actor 只在世界模型 rollout 中** 驱动探索，降低「为了梯度乱抖关节」的真机风险面（仍以任务与重置逻辑为前提）。
- **可复现入口齐全：** [论文](https://arxiv.org/abs/2601.21363)、[项目页](https://lift-humanoid.github.io/)、[GitHub 仓库](https://github.com/bigai-ai/LIFT-humanoid) 形成 **方法—视频—代码** 闭环，便于和工业界 PPO 主栈对照实验。

## 核心原理
| 阶段 | 目标 | 要点（归纳） |
|------|------|----------------|
| **(i) 策略预训练** | 大规模仿真里学到 **可零样本迁移** 的 locomotion 策略 | **MuJoCo Playground** 上千并行环境；**JAX SAC**；**大 batch + 较高 UTD**；**非对称 actor–critic**（critic 用含特权信息的状态）；域随机化；报告 **单卡 RTX 4090 墙钟约一小时量级** 的训练叙事（任务与搜索预算以论文为准）。 |
| **(ii) 世界模型预训练** | 学到 **可 rollout** 且 **带接触/耗散残差** 的下一步动力学 | 预训练日志 **全量落盘**，策略收敛后 **离线** 训练；**Brax 可微刚体**（质量阵、科氏力、重力）+ **半隐式欧拉** + **PD 映射扭矩**；残差头预测 **等效外扰扭矩** 与 **预测方差**；特权状态含 **基座高度**（论文强调对人形稳定性关键）。 |
| **(iii) 联合微调** | 新环境 / 新任务上 **少样本** 改善 | 真机或目标仿真中 **确定性行为采集** → 更新 buffer → **世界模型与 SAC 策略交替优化**；**随机策略仅用于模型内想象轨迹**。 |

### 流程总览

```mermaid
flowchart TB
  subgraph p1 [阶段 i 仿真预训练]
    Sim1["MuJoCo Playground\n大规模并行环境"]
    SAC["JAX SAC\n大 batch / 高 UTD"]
    Log["Transition 日志落盘"]
    Sim1 --> SAC
    SAC --> Log
  end
  subgraph p2 [阶段 ii 世界模型离线拟合]
    WM["物理知情世界模型\nBrax 拉格朗日 + 残差扭矩/方差"]
    Log --> WM
  end
  subgraph p3 [阶段 iii 适应微调]
    Det["目标环境\n确定性均值动作采集"]
    Buf["Replay buffer"]
    Roll["模型内 SAC rollout\n随机探索仅在此发生"]
    Det --> Buf
    Buf --> WM
    WM --> Roll
    Roll --> SAC
    SAC --> Det
  end
  p1 --> p2
  p2 --> p3
```

## 源码运行时序图

基于 `bigai-ai/LIFT-humanoid`（`main` 分支）三阶段入口脚本与核心模块：`train_in_mujoco_playground.py` 在 MuJoCo Playground 中并行运行 JAX SAC 预训练并落盘 `policy*.pkl` 与 `buffer_data/`；`train_wm_from_file.py` 读取 `buffer_data/` 在 Brax 动力学下离线训练物理知情世界模型并导出 `wm_state*.pkl`；`finetune.py` 加载两份状态后调用 `world_model/finetune_wm_ac.py` 执行「确定性环境采样进 `env_buffer` → 更新世界模型 → `ModelEnv` 内随机 rollout 进 `sac_buffer` → 混合两 buffer 更新 SAC」的交替微调闭环。

```mermaid
sequenceDiagram
  autonumber
  participant P1 as train_in_mujoco_playground.py
  participant SAC as policy_pretrain/train.py
  participant P2 as train_wm_from_file.py
  participant WM as world_model/pretrain_wm.py
  participant P3 as finetune.py
  participant FT as world_model/finetune_wm_ac.py
  participant BraxEnv as Brax 目标环境与 ModelEnv

  P1->>SAC: 加载 MuJoCo Playground 环境与 pretrain_sac_config
  SAC-->>P1: 输出 policies/policy.pkl 与可选 buffer_data
  P2->>WM: 加载 buffer_data 与 Brax 环境配置
  WM-->>P2: 离线训练残差世界模型并保存 wm_states/wm_state.pkl
  P3->>FT: 载入 sac_ts 与 wm_ts 及 dynamics_fn / reward_fn
  FT->>BraxEnv: prefill_env_buffer 用确定性动作采集真机或目标域轨迹
  loop 每个微调 epoch
    FT->>FT: wm_losses 更新物理知情世界模型参数
    FT->>BraxEnv: 在 ModelEnv 中以随机 SAC actor rollout 填充 sac_buffer
    FT->>FT: policy_update 按 real_ratio 混合 sac_buffer 与 env_buffer 更新 SAC
  end
```

- **最短复现路径：** `python train_in_mujoco_playground.py --env_name=T1LowDimSimFinetuneJoystickFlatTerrain --save_buffer_data` → `python train_wm_from_file.py --data_path=...` → `python finetune.py --ac_training_state_path=... --wm_training_state_path=...`；零样本部署前亦可用 `eval_in_brax.py` 或 `mujoco_env/play_mujoco.py` 转换并验证策略。
- **已核查仓库文件（`bigai-ai/LIFT-humanoid` @ `main`）：** `README.md`、`train_in_mujoco_playground.py`、`train_wm_from_file.py`、`finetune.py`、`world_model/pretrain_wm.py`、`world_model/finetune_wm_ac.py`、`eval_in_brax.py`、`policy_pretrain/train.py`。

## 常见误区或局限

- **误区：「SAC 不适合大规模并行仿真」。** LIFT 的论点是：在 **特定并行度、域随机化与调参（含 Optuna）** 前提下，SAC 可以 **稳定收敛** 并承担 **后续 model-based 微调** 的 **同一算法 backbone**；换任务与实现栈仍需重调，不宜外推为普遍结论。
- **误区：「零样本一定过 sim2real」。** 项目页给出反例叙事：**削弱能量约束的平地预训练** 可导致 **零样本实机失败**，再依赖 **分钟级** 微调恢复；说明 **奖励与地形分布** 仍是迁移主因，模型探索布局不能替代 **合理的预训练任务设计**。
- **局限：** 世界模型仍受 **残差容量与分布外误差** 约束；**Brax ↔ MuJoCo** 对齐、接触建模差异会占用大量工程精力（论文讨论状态映射与实现修正）。

## 工程实践

- **确定性采数 + 模型内探索分离：** 为防止真机随机动作探索摔机，LIFT 在真实机器人或高保真目标域只按确定性策略收集少数安全轨迹，将全部随机策略更新限制在物理残差世界模型内部完成，并在 Brax ↔ MuJoCo 状态对齐时校验关节零点顺序。

## 关联页面

- [Locomotion（运动任务）](../tasks/locomotion.md) — 人形行走 **学习路线与世界模型** 总览
- [Reinforcement Learning](../methods/reinforcement-learning.md) — SAC / model-based 分类坐标
- [Model-Based RL](../methods/model-based-rl.md) — Dreamer、MBPO 等对照语境
- [Sim2Real](../concepts/sim2real.md) — 域随机化、sim2sim、实机微调的安全与数据效率讨论
- [RL 算法选型（Query）](../queries/rl-algorithm-selection.md) — 足式场景下 PPO / SAC / TD3 的工程经验锚点
- [mjlab_playground](./mjlab-playground.md) — 与 **MuJoCo Playground 任务端口** 相关的训练生态交叉引用

## 参考来源

- [LIFT 论文摘录（arXiv:2601.21363）](../../sources/papers/lift_humanoid_arxiv_2601_21363.md)
- [lift-humanoid.github.io 项目页归档](../../sources/sites/lift-humanoid-github-io.md)
- [bigai-ai/LIFT-humanoid 仓库归档](../../sources/repos/bigai-lift-humanoid.md)
- Huang et al., *Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control*, [arXiv:2601.21363](https://arxiv.org/abs/2601.21363)

## 推荐继续阅读

- [机器人论文阅读笔记：LIFT](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/LIFT__Towards_Bridging_the_Gap_between_Large-Scale_Pretraining_and_Efficient_F/LIFT__Towards_Bridging_the_Gap_between_Large-Scale_Pretraining_and_Efficient_F.html)
- [DreamerV3（世界模型通用化）](https://arxiv.org/abs/2301.04104) — 潜空间想象主线的对照基准
- [MuJoCo Playground](https://playground.mujoco.org/) — 预训练任务与交互环境官方入口
- [Brax（JAX 可微物理与 RL 训练）](./brax.md) — 论文中 **可微刚体 rollout** 依赖的后端之一；维护边界以仓库 README 为准
