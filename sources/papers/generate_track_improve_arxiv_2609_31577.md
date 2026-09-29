# Generate, Track, Improve: Perceptive Multi-Skill Humanoid Locomotion with RL-Fine-Tuned Motion Generators（arXiv:2609.31577）

> 来源归档（ingest · 全文消化）

- **标题：** Generate, Track, Improve: Perceptive Multi-Skill Humanoid Locomotion with RL-Fine-Tuned Motion Generators
- **类型：** paper / perceptive humanoid locomotion / flow matching + CLF-RL tracking + off-policy generator fine-tuning
- **arXiv abs：** <https://arxiv.org/abs/2609.31577>
- **arXiv HTML：** <https://arxiv.org/html/2609.31577v1>
- **PDF：** <https://arxiv.org/pdf/2609.31577>；项目页镜像 <https://zolkin1.github.io/generate-track-improve/paper/generate-track-improve.pdf>
- **项目页：** <https://zolkin1.github.io/generate-track-improve/>（归档见 [`sources/sites/generate-track-improve-github-io.md`](../sites/generate-track-improve-github-io.md)）
- **代码：** **待发布** — 项目页 HTML 注释预留 Code 按钮，截至 2026-09-29 无 GitHub 链接
- **作者：** Zachary Olkin, William D. Compton, Aaron D. Ames
- **机构：** 加州理工学院（Caltech）Department of Control and Dynamical Systems / AMBER Lab
- **资助：** Technology Innovation Institute (TII)
- **硬件：** Unitree G1；ZED X + ZED X Mini 双深度；Jetson Thor（策略）+ Jetson Orin（深度）
- **入库日期：** 2026-09-29
- **一句话说明：** **Generate–Track–Improve** 管线：人类数据经 MuJoCo 多重打靶优化 + Motion Bricks 得 1 万条地形一致 clip → **CLF-RL 感知跟踪器**（50 Hz）+ **flow matching Transformer 生成器**（每 0.24 s 规划 1.24 s 全身轨迹，双 raw depth）→ **AWR 离线 RL** 微调生成器（结构化搜索 conditioning/初始噪声，非 2728 维高斯动作噪声）；G1 真机走/跑/跳箱/多级楼梯，成功率最高 +25 pp、技能选择 +80 pp。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://zolkin1.github.io/generate-track-improve/> | 真机/仿真视频、AWR 消融、双相机 Table IV |
| 同组跑步 | [Chasing Autonomy（2603.25902）](chasing_autonomy.md) | 动态优化参考 + CLF-RL；Olkin 前作 |
| 扩散+跟踪对照 | [ETH G1（2604.17335）](../papers/eth-g1-diffusion.md) | 地形条件扩散 + RL 跟踪 + 闭环 tracker 微调 |
| 深度行走 | [RPL（2602.03002）](rpl_arxiv_2602_03002.md) | 多视角深度 DAgger；无生成层 |
| 生成器 RL 对照 | AWR [48]、DPPO [43]、Residual off-policy [42] | 论文 §III-B 算法对比 |

## 摘要级要点

- **痛点：** 多技能 + 感知 + 动态 + 户外部署常分裂为「只 playback 的 mimic」与「只前进的深度地形策略」；生成–跟踪架构可模块化扩技能，但生成器在 OOD 几何上 mode 选择与地形一致性差。
- **架构：** 两层均 **raw depth**（无里程计/高程图）；速度指令可与地形解耦，机器人自行减速过障。
- **数据：** BONES-SEED 人类数据 → 252 平地稳态 + 46 跳箱/楼梯参考 → MuJoCo 多重打靶（周期 + 平均速度约束）→ 140 tile × 1 万 clip；**Motion Bricks** 做 in-between（相对 motion matching **>7×** 更快）。
- **跟踪：** CLF-RL 变体跟踪 **link 位姿**；Isaac Lab 8192 env × H100 ~48 h；30×26 下视深度 + 参考 MLP + 本体 MLP。
- **生成：** Flow matching Transformer；conditioning：双深度 + 速度 + 上一动作；输出 ~**2728** 维/计划。
- **微调：** Rollout（扰动 conditioning、重采样 flow 初始噪声、非常态 spawn）→ fitted value iteration critic → **AWR** 加权 flow MSE；**5 iter × ~6 h / H100**；相对全维 **PPO residual** 用 **>160×** 数据、**30×** 墙钟仍低 **~13 pp** 成功率。

## 核心摘录（面向 wiki 编译）

### 1) 部署时序（§II-A）

- Tracker **50 Hz**；Generator 每 **12** 控制步（**0.24 s**）重规划 **1.24 s** 全身轨迹。
- Tracker：本体 + **单路**下视深度；Generator：**双路**深度 + 速度 + 前一动作。

### 2) RL 微调奖励（§II-E · 归纳）

- 优先 **地形一致性**（SDF 穿透/接触罚）；速度跟踪与成功项在 plan 地形一致后才计分。
- 探索：禁止对 2728 维输出独立高斯噪声；改扰动 **conditioning**（存 nominal 为 label）、重采样 flow 噪声、非常态初始状态。

### 3) 实验数字（§III · 项目页 Table III–IV）

**Table III（clip 速度 RMSE，平地 vx 区间示例）：** 优化参考 + Motion Bricks 在中高速段 RMSE 显著低于单用 Motion Bricks（如 (+1.0,+2.5] m/s：0.442 vs 0.896 m/s）。

**Table IV（预训练生成器，分布内地形成功率）：**

| Terrain | Both cameras | Lower only |
|---------|--------------|------------|
| Flat | 100% | 100% |
| Box | 100% | 57.3% |
| Stairs up | 82.7% | 71.3% |
| Stairs down | 97.3% | 88.0% |

- **AWR vs 基线（Fig. 5）：** 优于 filtered BC、arrival filtering、plain BC/self-distill、无 conditioning 扰动的 AWR；plain BC 失败说明需要 advantage 筛选信号。
- **宏观增益（摘要/结论）：** 成功穿越最多 **+25 pp**；全技能正确选择最多 **+80 pp**（OOD 楼梯技能选择 0%→~80% 量级案例）。

### 4) 开源状态（项目页核查 · 2026-09-29）

- **待发布** — 无 Code 按钮；勿按 PDF 臆断已开源。

## 对 wiki 的映射

- 新建 [`wiki/entities/paper-generate-track-improve.md`](../../wiki/entities/paper-generate-track-improve.md)
- 交叉更新 [`wiki/methods/chasing-autonomy-pipeline.md`](../../wiki/methods/chasing-autonomy-pipeline.md)、[`wiki/tasks/humanoid-locomotion.md`](../../wiki/tasks/humanoid-locomotion.md)
