---
type: entity
tags: [paper, fudan, vla, active-perception, manipulation, imitation-learning]
status: complete
updated: 2026-10-20
project_id: activealign-observation-aligned-data-adaptation
arxiv: "2610.23378"
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../concepts/embodied-perception-six-spatial-representations.md
  - ../entities/paper-notebook-emergent-active-perception-and-dexterity-of-simu.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/papers/activealign_arxiv_2610_23378.md
summary: "ActiveAlign 通过闭环对齐主动视角轨迹终点与既有示范，并用成功续接片段微调 VLA；在 40 个 LIBERO 任务上 π₀/π₀.₅ 成功率分别为 70.7%/72.7%，实物遮挡测试由 35/90 提升到 83/90。"
---

# ActiveAlign：主动感知后的 VLA 观测对齐

**一句话定义：** ActiveAlign 在 VLA 前增加主动相机视角选择，并用从该视角终点成功续接到的示范数据微调 VLA，使策略适应相机动作改变后的首帧观测与机器人状态。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 接收视觉与语言指令并输出机器人动作的策略 |
| AP | Active Perception | 机器人主动调整视角以获得更有用观测 |
| LIBERO | Lifelong Benchmark for Robot Manipulation | 论文评估的多任务仿真操作基准 |
| BC | Behavior Cloning | 以示范动作监督微调策略的行为克隆方式 |

## 为什么重要

主动观察与物体操作常被当成两个可直接串联的模块，但相机移动会改变机器人姿态、目标在画面中的位置以及 VLA 看到的首帧。若操作策略只在常规示范起点训练，获得更好视角反而可能让它从陌生状态开始，形成“看得更清楚、动作却更差”的分布错配。

## 核心原理

在扰动初始状态的 LIBERO 测试中，直接在 π₀.₅ 前添加相机重定位会把平均成功率从 46.3% 降至 11.5%。ActiveAlign 将相机轨迹终点当作新的操作起点：离线闭环对齐到既有示范状态，重放其后续操作轨迹，筛选成功续接片段并用于策略微调。部署时先执行视角模块，再调用微调后的原架构 VLA。

### 数据适配闭环

```mermaid
flowchart LR
  DEMO[既有操作示范]
  VIEW[主动视角模块生成相机轨迹]
  END[视角轨迹终点状态]
  ALIGN[闭环对齐到示范状态]
  REPLAY[重放后续操作并筛选成功续接]
  FINETUNE[VLA 微调]
  DEPLOY[部署：先观察再操作]
  DEMO --> ALIGN
  VIEW --> END --> ALIGN
  ALIGN --> REPLAY --> FINETUNE --> DEPLOY
```

训练中的关键不是缩短动作前缀，而是让操作策略训练时见到由视角选择实际产生的观测和机器人状态。论文通过匹配状态与匹配数据量对照检验这一区分。

## 工程实践

| 环节 | 方法要点 | 实施关注 |
|------|----------|----------|
| 主动观察 | 视角模块产生相机运动轨迹 | 目标可见不等于到达状态符合 VLA 的训练分布 |
| 数据构建 | 将视角终点闭环对齐至既有示范并回放后续动作 | 对齐成功率和续接质量决定可用微调数据量 |
| 策略微调 | 用成功的续接操作轨迹适配 VLA | 可保留原模型架构，避免部署链路重构 |
| 评测 | LIBERO 40 项任务、遮挡实物测试、跨模型验证 | 对比应控制数据量和起始状态，排除简单前缀缩短解释 |

## 实验与评测

- π₀ + ActiveAlign 在 40 个 LIBERO 任务上为 70.7%，比标准训练基线高 40.0 个百分点；π₀.₅ 为 72.7%，高 26.4 个百分点。
- 物理遮挡任务中成功次数从 35/90 提升至 83/90。
- OpenVLA 和 GR00T N1 的实验呈现相同改善趋势；这一结果支持方法可迁移到不同 VLA，但不等于所有模型和任务都已验证。

## 结论

**一句话总判：** 主动感知应改变并适配策略训练起点；相机动作产生的新状态分布本身就是 VLA 训练数据设计的一部分。

1. 直接把相机动作当作固定前缀可能大幅损害成功率，首帧分布错配是必须单独测试的失败模式。
2. ActiveAlign 的核心资产是“视角终点 → 示范续接”的数据构造过程，而不是更换 VLA 架构或单纯扩大视角数据。
3. LIBERO 成功率改善幅度大，但应同时看论文中的匹配状态、匹配数据量对照，避免把收益误归因于更短的动作序列。
4. 实物遮挡测试从 35/90 到 83/90，提供了仿真外证据；仍需在更多真实机器人、摄像机安装方式和任务中检验。
5. 截至 2026-10-20，arXiv 页面未列项目页或官方代码仓库，复现数据适配和部署仍受公开实现缺失限制。

## 与其他工作对比

| 方案 | 视角变化后的操作起点 | 主要风险 |
|------|--------------------|----------|
| 直接串接 AP 与现成 VLA | 使用原 VLA，未适配的新视角终点 | 首帧与状态分布外，可能性能骤降 |
| 通用视角增强 | 训练中增加不同视图 | 不一定覆盖主动相机运动后实际到达的机器人状态 |
| ActiveAlign | 用闭环匹配与成功续接轨迹微调 | 需要可靠对齐流程及足够成功续接数据 |

## 局限与风险

方法依赖可用于闭环对齐的示范轨迹及视角终点后的成功续接数据。若视角模块进入示范覆盖范围之外，或对齐错误，微调数据可能无法支持后续操作。论文为 2026 年 arXiv 预印本；当前来源未提供公开代码仓库，具体对齐算法、训练配置与部署细节需以全文为准。

## 关联页面

- [VLA](../methods/vla.md) — 视觉语言动作策略基础
- [Manipulation](../tasks/manipulation.md) — 操作任务与基准语境
- [Embodied perception：六类空间表示](../concepts/embodied-perception-six-spatial-representations.md) — 主动视觉和空间观测
- [Emergent Active Perception and Dexterity](../entities/paper-notebook-emergent-active-perception-and-dexterity-of-simu.md) — 主动感知与操作能力的相关研究
- [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md) — 本页感知能力在机器人视觉感知栈中的位置与选型

## 参考来源

- [论文来源归档](../../sources/papers/activealign_arxiv_2610_23378.md)
- [论文 HTML 全文](https://arxiv.org/html/2610.23378v1)

## 推荐继续阅读

- [Look Before You Act / ActiveAlign（arXiv:2610.23378）](https://arxiv.org/abs/2610.23378)
- [VLA](../methods/vla.md)
- [Emergent Active Perception and Dexterity](../entities/paper-notebook-emergent-active-perception-and-dexterity-of-simu.md)
