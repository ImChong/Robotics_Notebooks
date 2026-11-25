---
type: entity
project_id: partymimic
project: https://roboparty.github.io/PartyMimic/
code: https://github.com/Roboparty/PartyMimic
tags: [humanoid, motion-tracking, model-based-rl, imitation-learning, sim2real, roboparty, open-source]
status: complete
updated: 2026-11-25
related:
  - ./mimiclite.md
  - ./blinkmimic.md
  - ./cn-os-xmimic.md
  - ./party-os.md
  - ../concepts/motion-retargeting.md
  - ../concepts/sim2real.md
  - ../methods/model-based-rl.md
  - ../methods/beyondmimic.md
  - ../../roadmap/depth-bfm.md
sources:
  - ../../sources/sites/partymimic.md
  - ../../sources/repos/partymimic.md
original:
  - ../../sources/sites/partymimic.md
  - ../../sources/repos/partymimic.md
summary: "PartyMimic（RoboParty Lab）是基于在线可微世界模型与32步 BPTT 的单卡分钟级人形全身动作跟踪方法，覆盖 G1/RP1、MLP/Attention 和完整关节/稀疏姿态两种参考接口；公开数据配方、训练代码、ONNX 与 MuJoCo sim2sim 评测。"
---

# PartyMimic — 单卡分钟级人形全身运动跟踪 / General Humanoid Motion Tracking in Minutes

## 一句话定义

**PartyMimic** 是 RoboParty Lab 在 SuperMimic / BlinkMimic 技术线上发展的可微世界模型全身动作跟踪方法：从多来源动作库训练一份人形策略，通过 32 步想象展开的梯度反传优化跟踪表现，并开放完整训练、数据处理、导出及评测入口。项目方报告在 **单张 RTX 4090** 上，以 MLP 对约 **41 小时**参考动作训练 12,000 次更新，G1 **28 分钟**、RP1 **29 分钟**。这是项目页实验配方，不等于当前部署默认参数或随仓权重的单次测量。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| WM | World Model | 从仿真状态转移拟合、持续在线更新的动力学预测模型 |
| BPTT | Backpropagation Through Time | 对 32 步可微想象轨迹反向传播策略梯度 |
| MLP | Multi-Layer Perceptron | 默认目标参考编码器，可在 24 GB 4090 运行 |
| EMA | Exponential Moving Average | 对策略参数滑动平均稳定训练 |
| IK | Inverse Kinematics | 使用 G1 到 RP1 的机器人间参考动作重定向 |
| ONNX | Open Neural Network Exchange | 模型导出并交由外部 sim2real 评测器执行 |

## 为什么重要

- **算力门槛：** 官方在相同的 10,745 条 / 41 小时动作库上，MLP 的 G1/RP1 峰值显存分别为 22.6/23.4 GB；完整 cross-attention 配置均约 43 分钟，通常超出 24 GB。
- **明确的泛化设置：** 在 Loco（80）、Mani（48）、Dance（40）、Ground（60）和 Extreme（308）五个未见动作集评测；Extreme 按源视频划分，降低同源片段泄漏。
- **接口可复用：** `joint` 使用根部运动与关节参考；`sparse5` 使用骨盆及头、两腕、两踝共六组参考姿态，不依赖特定机器人关节编号，方便跨人体/机器人输入。
- **配方与负结果公开：** 既公布有效优化，也展示 SuperMimic 多因素消融的无明显作用和失败设置。

## 核心原理

### 1. 数据—训练—评测流程总览

```mermaid
flowchart TD
  A["多源动作库: LAFAN1 / BONES-SEED / KungfuAthlete 等"] --> B["筛选多样性与动作质量"]
  B --> C["50 Hz 运动库: G1 与 RP1 IK 重定向"]
  C --> D["mjlab / MuJoCo Warp 批量仿真"]
  D --> E["ReplayBuffer 采集真实仿真转移"]
  E --> F["在线拟合 WorldModel"]
  E --> G["采样想象起点"]
  F --> H["WorldModel 内 32 步想象展开"]
  G --> H
  C --> I["参考目标编码: MLP 或 Cross-Attention"]
  I --> H
  H --> J["跟踪奖励 + 终端 Critic V"]
  J --> K["BPTT 更新 Policy + Muon/EMA"]
  H --> L["GAE 拟合 Critic + value-gradient consistency"]
  K --> D
  L --> J
  K --> M["checkpoint → ONNX + YAML"]
  M --> N["MuJoCo sim2sim: 五组未见动作"]
```

模型从 `TrackingSim` 中采集状态转移，`WorldModel` 拟合动力学（包含对控制的可微依赖）。`Policy` 通过网络预测关节控制参考，`objective.actor_objective` 组合 32 步跟踪奖励、存活项与终端 Critic 价值；反向传播通过未来观测反馈链路回到策略。Critic 依据想象轨迹的 GAE 目标及可选值梯度一致性损失拟合。重要区别是**策略不是仅用一次模型前向预测优化，也不冻结世界模型**。

### 2. 训练配方与超参

| 机制 | 官方默认 / 开关 | 项目页消融（去掉该项后） |
|---|---|---|
| Muon 优化策略与 WM | `--pi_opt muon --wm_opt muon` | G1 Loco/Mani 误差 +10.7 mm |
| 缩窄跟踪奖励核 | `--std_scale 0.35` | +6.5 mm |
| 想象时域 | `--horizon 32` | 改 16 步：+5.9 mm |
| 参考动作残差 | `--pi_ref_skip` | +5.6 mm |
| 策略滑动平均 | `--pi_ema 0.995` | +3.4 mm |
| 值梯度一致性 | `--v_gcons 0.5` | +1.0 mm |
| 多样性均衡重置 | `--div_k 24` | Ground +6–19 mm |

> 消融数字来自项目页较早的 **9,173 clips** 版本（G1 Loc/Man 口径）；不能与最终 10,745 条 held-out 配方的结果表直接视作同一次实验。

### 3. MLP、Attention 与稀疏姿态

- **MLP（默认）：** 编码多帧参考目标，训练时间/显存开销低；G1 五组平均误差 38.4 mm。
- **Cross-attention：** 由观测历史查询未来目标，G1 五组平均误差 37.4 mm，完整配置约 43 分钟；24 GB 可减小 `--pi_batch`，但不宜套用完整配置的表格结果。
- **`sparse5`：** 参考包含 pelvis + head + 两个 wrist + 两个 ankle **六组身体姿态**；使用当前机器人正向运动学得到关键点误差，预拟合并冻结 `kp2q` 用于参考动作 skip，辅助头预测参考关节角。RP1 MLP 五组平均误差 42.3 mm（完整关节参考 39.3 mm）；ONNX 原始 `ref_kp` 是每帧六组 `xyz + wxyz` 姿态，和网络内部特征维度分开记。
- 命名里的 “5” 表示骨盆之外的五个部位，不是仅五个输入位置，也不等价于仅 5 个关节角。

## 实验与评测

项目方在 2 个种子、通常 12,000 次更新下，使用 EGalahad 的 MuJoCo sim2sim 评测器跑导出后的 ONNX。下表是按相对骨盆航向对齐、**已跟踪片段**的身体位置误差；五组先各自求均值，再等权汇总。Extreme 严格成功率另算，要求从头到尾跟完，且骨盆姿态误差超 57° 不持续 0.2 秒。SONIC 和 Mimic Lite v1.1 采用已有权重，并非同卡重训；NablaMimic 保留两小时预算。

| 机器人 / 方法 | Mani 误差 | 五组平均误差 | 训练时间 |
|---|---:|---:|---:|
| G1 PartyMimic MLP | 21.4 mm | 38.4 mm | 28 min |
| G1 PartyMimic attention | 21.7 mm | 37.4 mm | 43 min |
| G1 BlinkMimic | 30.6 mm | 53.5 mm | 59 min |
| G1 SuperMimic（官方源码重接 mjlab） | 37.2 mm | 57.5 mm | 99 min |
| RP1 PartyMimic MLP | 16.3 mm | 39.3 mm | 29 min |
| RP1 PartyMimic attention | 16.5 mm | 38.7 mm | 43 min |
| RP1 PartyMimic sparse5 MLP | 19.2 mm | 42.3 mm | 31 min |

这组已更新的 10,745 条训练参考包含旧 10,731 条配方加 14 条 XtremeMotion。旧页的 26 分钟、G1 Mani 20.5 mm 与该表不是同一快照，不能直接算成版本净增益。随仓 G1/RP1 MLP 权重为 seed 0，README 单独报告五组均值 **39.0 / 39.6 mm**，不是表中双种子均值。

**LAFAN-40 单独指标：** 3–5 分钟是同一批 40 条 LAFAN1 **训练兼评测**样本的 return/零失败学习速度测试；单卡返回 ≥0.82 首次用 2.45 分钟，连续三次评测无失败需 5.17 分钟（加 warm start 5.57 分钟）。不能直接解释成 41 小时多类别 zero-shot 泛化仅用 3–5 分钟。

### SuperMimic 消融最有用的结论

- 多步时域梯度或在线 WM 拟合不可轻易省略：仅保留单步模型梯度得分约 0.002；1000 次迭代后冻结世界模型得分约 0.004（基线 0.686±0.048）。
- 没有终端 Critic 约 0.428，H=8 约 0.430；过短时域让策略过度依赖不精确的值梯度。
- 取消后续观测对状态的反馈梯度得分约 0.517；对闭环控制尤其不利。
- 关闭自碰撞在部分训练评估上更好，但切换 leaderboard 碰撞模型后 40 条 LAFAN1 里有 33 条跌倒：必须匹配部署评测的碰撞条件。
- 单项消融包含噪声和多个单 seed 变体，不应把“趋势”当作显著性结论。

## 与其他工作对比

- **SuperMimic / BlinkMimic / XMimic：** 主表中 PartyMimic 重训在同一多源库上的五组均值更低，但早停标准不同会反转排序。LAFAN-40 同集条件下，单卡 PartyMimic 达到 0.82 return 用 2.45 分钟，连续三次零失败需 5.17 分钟；XMimic 对应为 2.32 / 3.71 分钟（均不含启动）。这组数据不回答未见动作泛化问题。
- **与实机证据分开：** 官方有 G1/RP1 真机片段；延迟抖动表、批量过场统计，以及 VLM 把箱子搬到放置区的 14/20，是各自的仿真实验，不能把它们当成真机成功率。项目页另报 RP1 低箱抓取：关键帧直接追踪始终未提起箱子；加逐帧全身 IK 后，MuJoCo 的 216 次试验中约 59% 提起、3% 跌倒。箱体靠摩擦接触而非焊接；这是规划/控制接口对照，不是随仓权重在实机上的承载指标。

## 工程实践

```bash
git clone https://github.com/Roboparty/PartyMimic.git
cd PartyMimic
uv sync
tools/download_data.sh
tools/data/build_library.sh g1
bash scripts/train.sh logs/g1_s0 --robot g1mj --motion_dir data/motions/g1_mix --seed 0
# 可选: --ref_enc xattn 或 --interface sparse5
# 导出:
mkdir -p deploy
uv run python tools/export_deploy_onnx.py checkpoints/g1_mlp.pt deploy/g1_mlp.onnx
# sim2sim 需额外安装 EGalahad/sim2real、应用 tools/sim2real.patch 并提供其目录
```

- **软件 / 资源：** 官方说明 Linux x86_64、CUDA 12.8、`uv`；完整 41 h 动作库依赖 BONES-SEED（需接受 gated 数据授权）。`--num_envs 4096`、`--hist 10`、`--goal_frames 10`、`--horizon 32` 为训练入口默认值。
- **训练配方不是一个名字：** CLI 默认在物理步长 5 ms 下随机保留上个关节目标 1–4 个子步，首个控制步不延迟；不开项目页的 clip allocation 或联合口径 `--obs_err kp`。项目页主表及发布 MLP 权重没有这项命令滞后。RP1 Mani 在动作晚 10–20 ms、观测旧 5–10 ms 的仿真下，主表配方 → 单加滞后 → 完整部署默认，脚滑为 3.75 → 1.42 → 1.03 cm/s；末次变更还删了两项配置，不能全归因于滞后。
- **地形分支：** `--terrain` 首次可从参考动作推断并检查支撑盒，失败片段排除，仿真重置放置地形；高度扫描给世界模型和 Critic，不进 Actor。G1 过滤后的 BONES-SEED 为 129,205 条，3,000 次更新时全库成功 99.05%、有地形子集 89.1%；这些是训练库回放。该大库地形配方需约 26.3 GB 显存，不能套用 24 GB 的综述宣传。
- **RP1：** 运行 `tools/data/build_library.sh rp1` 需 human-humanoid-tools（官方固定提交），用机器人间 IK 转成 RP1 28 DoF 运动库。
- **模型导出：** `tools/export_deploy_onnx.py` 构建不同接口的 ONNX 与部署 YAML，稀疏输入的 ONNX 参数包含 `ref_kp`；随后通过外部 `EGalahad/sim2real` 测试。
- **两卡流水线：** `--pipeline` 使 cuda:0 收集与拟合 WM，cuda:1 处理上一轮数据/模型的策略和 critic 更新，通过 pinned-memory 异步传送；不同于默认单卡训练。

## 源码运行时序图

以下依据官方 `scripts/train.sh`、`train.py` 中 `collect` / `wm_update` / `policy_step` / `critic_fit`、`objective.py` 及 `scripts/eval.sh` 绘制；展示默认单卡顺序，未把可选两卡流水线误写为默认路径。

```mermaid
sequenceDiagram
  participant Shell as scripts/train.sh
  participant Main as partymimic.train
  participant Sim as TrackingSim
  participant Buf as ReplayBuffer
  participant WM as WorldModel
  participant Pi as Policy
  participant Obj as actor_objective
  participant V as Critic
  Shell->>Main: python -m partymimic.train
  opt --terrain 且无 terrain.json
    Main->>Main: terrain_infer.build 推断并过滤支撑几何
  end
  Main->>Main: MotionLib.from_dir + robot + networks
  Main->>Sim: set_act_lag 绑定训练命令时延
  loop training iterations
    Main->>Sim: collect 经 CUDA Graph 采集，step_core 保留旧目标直至滞后结束
    Sim-->>Buf: add_device(transition)
    Main->>Buf: sample world-model windows
    Buf-->>WM: fit transition dynamics via wm_update
    Main->>Buf: sample_valid_starts(pi_batch)
    Main->>Pi: policy_step: 32 imagined steps
    Pi->>WM: imag_step state transition
    WM-->>Pi: next imagined state / observation
    Pi->>Obj: reward + survival + V terminal
    Obj->>V: terminal state value
    V-->>Obj: bootstrap value
    Obj-->>Pi: BPTT actor loss gradient
    Main->>Pi: optimizer step + EMA
    Main->>V: critic_fit on imagined GAE targets
  end
  Main->>Main: write checkpoint
  Note over Shell,Main: 训练结束后独立运行 export_deploy_onnx.py 输出 ONNX/YAML
```

## 结论

**选型总判：PartyMimic 证明在项目方统一的多源未见动作评测协议里，可微模型 BPTT 加工程配方能在单卡内提高全身跟踪效果，但不能把所有速度和跨基线结果都解读成严格同协议独立复现。**

1. 24 GB 单卡优先默认 MLP；追求极限误差且显存允许再尝试 attention。
2. 想接人类动作、VR 追踪或跨机器人命令，可用 `sparse5`；会增加 `kp2q` 等准备步骤。
3. 保持 WM 在线拟合、32 步闭环梯度、策略 EMA/参考 skip，先匹配部署端碰撞模型再调奖励。
4. 对 Ground/Extreme 的性能同时查看误差和完成率，并检查单目动作的滑脚与执行器速度上限。
5. 比较训练耗时时区分 41 h 跨源训练与 LAFAN-40 同集速度基准，以及同卡重训和官方现成策略。

## 局限与风险

- **数据授权：** 仓库代码 MIT 不代表数据或衍生 checkpoint 全部 MIT：LAFAN1 CC BY-NC-ND，MotionDecode 非商业条款，BONES-SEED 另需授权。
- **非独立复现：** 本次静态核查项目页、README 与固定提交的关键训练/导出源码；未执行 4090 训练、ONNX 推理或真机实验，所有性能指标均明确为官方结果。
- **真实运动极限：** Extreme 最快 1/3 动作的 G1 身体误差约 77 mm（最慢组 22 mm），单目参考脚底滑动可达约 0.98 m/s，快速摆动可能超出执行器能力。
- **sim2real 边界：** 仓库给出 G1/RP1 仿真与外部部署导出路径；不能从 sim2sim 数值单独证明实机鲁棒性。
- **结果比较：** SONIC/MimicLite 官方权重与本仓重训基线训练资源/数据不相同；Project 页对 “best” 的说法仅在列出的对比方法与协议内成立。

## 关联页面

- [MimicLite：RoboParty 另一套跟踪基础设施](./mimiclite.md)
- [BlinkMimic：快速模仿训练前序实现](./blinkmimic.md)
- [XMimic：JAX 快速训练对照](./cn-os-xmimic.md)
- [Party OS / RoboParty Lab](./party-os.md)
- [动作重定向 / Motion Retargeting](../concepts/motion-retargeting.md)
- [Sim2Real](../concepts/sim2real.md)
- [Model-Based RL](../methods/model-based-rl.md)：区分在线拟合动力学与在模型中更新策略。
- [BeyondMimic](../methods/beyondmimic.md)：对照参考跟踪与部署评测的控制接口。
- [BFM 纵深路线](../../roadmap/depth-bfm.md)：Stage 2 全库动作跟踪。

## 参考来源

- [PartyMimic 项目页原文归档](../../sources/sites/partymimic.md)
- [PartyMimic 官方源码与 README 归档](../../sources/repos/partymimic.md)
- [官方项目主页（全文实验表、SuperMimic 消融）](https://roboparty.github.io/PartyMimic/)
- [官方 GitHub / README / 源码](https://github.com/Roboparty/PartyMimic)

## 推荐继续阅读

- [完整项目页与消融表](https://roboparty.github.io/PartyMimic/#ablation)
- [SuperMimic 原论文 arXiv:2610.28480](https://arxiv.org/abs/2610.28480)
- [PartyMimic GitHub](https://github.com/Roboparty/PartyMimic)

