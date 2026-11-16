---
type: entity
project_id: terrain-indexed-parkour
tags:
  - paper
  - humanoid
  - parkour
  - amp
  - perceptive-locomotion
  - noetix
status: complete
updated: 2026-11-09
arxiv: "2611.08056"
original:
  - ../../sources/papers/terrain-indexed-parkour_arxiv_2611_08056.md
related:
  - ../methods/amp-reward.md
  - ../tasks/humanoid-locomotion.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
  - ../entities/paper-hil-hybrid-imitation-learning.md
  - ../entities/paper-deep-whole-body-parkour.md
  - ../entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md
  - ../entities/paper-now-you-see-that-humanoid-vision-locomotion.md
  - ../entities/paper-light-loco-parkour.md
  - ../entities/cn-os-noetix-e1-lab.md
sources:
  - ../../sources/papers/terrain-indexed-parkour_arxiv_2611_08056.md
summary: "Terrain-Indexed Motion Priors and Geometric Support（arXiv:2611.08056）：特权模仿 → 地形索引 AMP → 降 AMP 权重加手/膝几何支撑，三阶段训练单一深度策略；Noetix E1 仿真混合赛道 96.34%，真机零样本 29/30 单障碍。"
---

# 面向极限人形跑酷的地形索引动作先验与几何支撑（arXiv:2611.08056）

**Terrain-Indexed Motion Priors and Geometric Support for Extreme Humanoid Parkour**（[arXiv:2611.08056](https://arxiv.org/abs/2611.08056)，2026-11-05，cs.RO）。作者 Chenghao Sun、Youhao Lyu、Zizheng Li、Shuting Zhang、Ziyang Dong、Wenhao Cui；机构（论文首页脚注）：松延动力 Noetix Robotics、中国科学技术大学、国防科技大学，其中 Zizheng Li 的机构字段在 v1 中仍是占位符。原文归档：[sources/papers/terrain-indexed-parkour_arxiv_2611_08056.md](../../sources/papers/terrain-indexed-parkour_arxiv_2611_08056.md)。

## 一句话定义

**一个三阶段渐进模仿框架：先用特权模仿把人类跑酷示范变成机器人可执行参考，再用地形索引 AMP 习得 Low / Mid / High 三类穿越技能，最后把 AMP 权重降到一半并加入手 / 膝几何支撑引导，让接触位置适应真实障碍表面；部署时单个策略只用板载深度、本体感觉与速度命令。**（作者自述，摘要与 Sec. III-A）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AMP | Adversarial Motion Priors | 对抗式运动先验：判别器区分参考与策略运动窗口，输出风格奖励（见 [amp-reward](../methods/amp-reward.md)） |
| PPO | Proximal Policy Optimization | 更新 actor、critic 与深度编码器的 on-policy 算法（Sec. III-A） |
| HIM | History Estimator | 论文中用 5 帧本体历史估计 3 维速度与 16 维隐变量，独立优化器训练（Sec. III-A） |
| RSI | Reference-State Initialization | 参考状态初始化，训练概率 0.8，评测关闭（Sec. III-B） |
| NMS | Non-Maximum Suppression | 候选去重，0.10 m，每肢每格最多留 32 个（Sec. III-C） |
| DTW | Dynamic Time Warping | 用关节角序列比较执行轨迹与 High / Mid / Low 参考（Sec. IV-B） |
| OOD | Out-of-Distribution | 障碍高度在训练范围外 1-3 cm（Sec. IV-A） |
| GAE | Generalized Advantage Estimation | 优势估计，λ=0.95（表 II） |

## 为什么重要

- 作者指出的痛点（Sec. I）：人类示范的动作与手 / 膝接触位置不一定适合机器人的身材、驱动与实际支撑面；同时映射动作可能违反机器人物理、可执行参考未必适合当前地形。
- 作者的设计原则是「把模仿当脚手架」：先习得技能，再放松先验、加强几何支撑（Sec. I）。
- 对本 wiki 读者：它把 [AMP](../methods/amp-reward.md) 的「多判别器 / 地形条件」路线与手 / 膝全身接触奖励结合，可与 [HIL](../entities/paper-hil-hybrid-imitation-learning.md)、[Perceptive Humanoid Parkour](../entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)、[Now You See That](../entities/paper-now-you-see-that-humanoid-vision-locomotion.md) 对照阅读。

## 核心原理与方法栈

### 流程总览

```mermaid
flowchart LR
  H["人类跑酷示范"] --> R["重定向 + Stage 1 特权模仿"]
  R --> P["机器人参考 rollout 含力矩 接触 高度扫描"]
  P --> A["Stage 2 地形索引 AMP 技能习得"]
  A --> S["Stage 3 降 AMP 权重 加几何支撑引导"]
  S --> D["部署策略 深度 本体 速度命令"]
  D --> E["Noetix E1 板载 Jetson Orin NX"]
```

### 1. 策略与信息边界（Sec. III-A）

- 非对称 actor-critic：actor 输入 78 维本体 / 命令向量、24×24 深度图（截断 0.3-4 m）；HIM 输出 3 维速度和 16 维隐变量，加 32 维深度特征，共 129 维 actor 输入；actor 与 critic 各 4 个门控专家，输出关节位置目标。
- critic 额外获得 4 维特权地形标签；真实地形标签、判别器高度扫描、支撑几何都只在训练期使用。
- 总奖励（Eq. 1）：`r_t = 0.5Δt [R_other + α(n) S_t] + β(n) φ_k(U_t)`，基础任务奖励沿用 legged gym 设计。

### 2. 地形索引 AMP（Sec. III-B）

- 判别器输入每帧 402 维：关节位置 23、本体位置 72、速度项、接触 24、力矩 23、高度扫描 231（Eq. 2），4 帧窗口。
- 共享主干（512、256）+ 4 个 256→1 判别头（High / Mid / Low / Flat，表 I）；站立与前行参考出现在多个池里；当前地形标签路由到对应头，给出风格奖励 φ_k（Eq. 3-6）。
- 8 个参考片段：335 站立、785 转向、735 侧移、770 后退、718 前行、174 / 136 / 180 帧的 Low / Mid / High 穿越（Sec. IV 训练细节）。

### 3. 几何手 / 膝支撑（Sec. III-C）

- 训练期表面候选图：0.025 m 采样，高度 0.30-1.10 m；手 / 膝足印 0.12×0.08 / 0.10×0.07 m；椭球可达性过滤，每肢保留至多 8 个候选，每 5 个控制步刷新。
- 预接触：以 0.12 s 预测末端位置，对候选做得分加权吸引（Eq. 8）；接触后用实测末端评估实际支撑；`S_place = .15 I_H + .35 A_H + .10 I_K + .25 A_K`（Eq. 9）。
- 支撑保持 / 惩罚项带阶段门控，不规定接触顺序。
- 先验放松：α 由 0.1 线性升到 1（4,000 次更新后到位，Eq. 10），AMP 权重 β 由 1 降到 0.5；候选奖励仅在 Stage 3 启用。

## 工程实践

- **训练栈：** Isaac Lab + 基于 RSL-RL 的 PPO；4,096 并行环境、24 步 rollout；三阶段迭代 20,000 / 20,000 / 10,000（表 II）。可对照 [Isaac Lab](../entities/isaac-lab.md)、[RSL-RL](../entities/rsl-rl.md)、[legged gym](../entities/legged-gym.md)。
- **部署：** Noetix E1（23 自由度、板载 IMU / 编码器 / 底座深度相机，光轴俯仰 60°），策略在 Jetson Orin NX 上运行并输出关节位置命令；真机前在 MuJoCo 与 Gazebo 做控制器验证（Sec. IV）。机器人平台背景见 [Noetix E1 实验室仓库](../entities/cn-os-noetix-e1-lab.md)。
- **评测口径：** 完整穿越成功（越过障碍、清出口、之后稳定支撑 1 s），不要求特定手 / 膝接触；单障碍限时 20 s，混合 60 s；按验证集选 checkpoint（Sec. IV-A）。

## 实验结果（论文报告，作者自述数字）

| 条件（表 III，2,048 episode） | Low | Mid | High | Hybrid | OOD |
|------|------|------|------|------|------|
| Ours | 100 | 98.44 | 98.14 | 96.34 | 93.46 |
| w/o Stage 2 | 41.36 | 21.14 | 13.87 | 0 | 0 |
| w/o Stage 3 | 100 | 85.55 | 79.69 | 83.59 | 85.11 |
| Velocity Tracking | 96.09 | 0 | 0 | 0 | 0 |
| Single AMP | 100 | 65.82 | 0 | 0 | 44.29 |
| Multi-Skill Distillation（本地实现，基于 PHP） | 100 | 99.28 | 95.10 | 79.65 | 65.33 |

- **支撑适应消融（表 IV）：** 1 m×1 m 顶面平台上 Ours 93.46%、w/o Candidates 46.24%、w/o Relaxation 58.89%、w/o Stage 3 50.63%；Low 几乎无差别（Sec. IV-C）。
- **时间维度对应（Sec. IV-B，Fig. 4）：** 一条成功轨迹上 DTW 最近参考依次为 High（1.50-4.88 s）→ Mid（4.88-8.00 s）→ Low（8.00-12.54 s），作者称这是定性证据。
- **真机（表 V）：** Low 10/10、Mid 9/10、High 10/10（共 29/30）、混合 4/5、OOD 13/15，零样本、无真机微调（Sec. IV-D）。

## 结论

**三阶段「模仿当脚手架」在仿真混合与 OOD 赛道上明显优于单 AMP 与多技能蒸馏基线，但各阶段贡献与训练预算混在一起，真机样本也很小。**

1. **真影响指标是 Hybrid 与 OOD 成功率**：单障碍上 Multi-Skill Distillation 与 Ours 接近，差距主要在 Hybrid（79.65 对 96.34）与 OOD（65.33 对 93.46）（表 III）。
2. **地形索引 AMP 是必要的技能来源**：w/o Stage 2 的 Hybrid / OOD 均为 0；Single AMP 在 High 与 Hybrid 也为 0。
3. **几何支撑主要在非标准顶面上起作用**：平台测试 w/o Candidates 46.24%、w/o Relaxation 58.89%（表 IV），而 Low 几乎无差别。
4. **次要但需自测的指标**：论文不主张更低接触力或更大稳定裕度，部署前需补接触力与安全测量。
5. **部署读法**：单平台 Noetix E1、真机每类 10 次以内、无置信区间；混合赛道的突然转向是已观察到的失败模式；无公开代码。

## 与其他工作对比

| 对照（表 III） | 路线 | Hybrid / OOD | 读法 |
|------|------|------|------|
| Velocity Tracking | 无动作先验的速度跟踪 | 0 / 0 | 只能过 Low |
| Single AMP | 单判别器 AMP，不按地形路由 | 0 / 44.29 | 地形索引多头判别器的必要性（但不能单独证明主干共享） |
| Multi-Skill Distillation | 作者基于 [PHP](./paper-hrl-stack-22-perceptive_humanoid_parkour.md) 的本地实现：多技能专家蒸馏为单一深度策略 | 79.65 / 65.33 | 单障碍接近，混合与 OOD 落后；不代表 PHP 原文性能 |
| **Ours** | 特权模仿 → 地形索引 AMP → 降先验 + 手 / 膝几何支撑 | 96.34 / 93.46 | — |

本页关联的 [HIL](./paper-hil-hybrid-imitation-learning.md)、[Deep Whole-Body Parkour](./paper-deep-whole-body-parkour.md)、[Now You See That](./paper-now-you-see-that-humanoid-vision-locomotion.md)、[Light-Loco-Parkour](./paper-light-loco-parkour.md) 同属人形感知跑酷方向，但论文未与之做实验对比，平台与评测口径不同，不能横比成功率。

## 局限与未验证之处

**论文自述的局限（Sec. IV、V）：**

- 去掉某阶段同时去掉了其训练预算，结果不能把各阶段目标与额外训练量分开；Stage 3 对照同样如此（Sec. IV-A、IV-C）。
- Single AMP 对比评估的是整套地形索引方案，不能单独证明主干共享，也不能证明避免了模式坍缩（Sec. III-B）。
- DTW 分析只用一条成功轨迹，不估计技能切换准确率，也无法区分「学到的切换」与「路线上的动作相似」（Sec. IV-B）。
- 结论不是更低接触力、更大稳定裕度或接触安全保证；需要成功率之外的测量（Sec. IV-A、IV-C）。
- 真机失败：Mid 一次手臂支撑丢失；混合一次因突然转向后出现 Mid 型翻转，说明支撑保持与地形匹配的技能转换仍是难点（Sec. IV-D）。
- 未来工作：扩大参考覆盖、为不规则表面生成支撑候选（Sec. V）。

**本页作者的观察（非论文原话）：**

- 论文只用单一机器人 Noetix E1；真机次数很少（单障碍每类 10 次、混合 5 次、OOD 15 次），未给置信区间；仿真「每个训练种子」的种子数在正文中没有写明。
- 表 IV 平台列的 Ours（93.46）与表 III 的 OOD 列 Ours（93.46）数值完全相同，可能是巧合也可能是抄录问题，原文未说明，待核实。
- 「极限」只体现在标题；文中最高障碍约 100 cm（Fig. 1 标注 100 / 66 / 36 cm），机器人身高未给出，本页不对「极限」程度作判断。
- Multi-Skill Distillation 基线是作者自己对 PHP 的本地实现，不代表 PHP 原文性能（Sec. IV-A 作者自己也注明）。
- 代码与项目页：论文未给出链接（入库日 2026-11-09）。

## 关联页面

- [AMP 奖励方法页](../methods/amp-reward.md)
- [人形运动控制任务页](../tasks/humanoid-locomotion.md)
- [楼梯 / 障碍感知运动](../tasks/stair-obstacle-perceptive-locomotion.md)
- [HIL：视频跑酷的混合模仿](../entities/paper-hil-hybrid-imitation-learning.md)
- [Deep Whole-Body Parkour](../entities/paper-deep-whole-body-parkour.md)
- [Perceptive Humanoid Parkour](../entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)
- [Now You See That](../entities/paper-now-you-see-that-humanoid-vision-locomotion.md)
- [Light-Loco-Parkour](../entities/paper-light-loco-parkour.md)
- [Noetix E1 实验室仓库](../entities/cn-os-noetix-e1-lab.md)
- [PPO](../methods/ppo.md)

## 参考来源

- [sources/papers/terrain-indexed-parkour_arxiv_2611_08056.md](../../sources/papers/terrain-indexed-parkour_arxiv_2611_08056.md)
- [arXiv:2611.08056](https://arxiv.org/abs/2611.08056)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.08056v1)
- [AMP 奖励方法页](../methods/amp-reward.md)
- [Perceptive Humanoid Parkour](../entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)
