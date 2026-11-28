---
type: concept
tags: [affordance, embodied-ai, manipulation, perception, grounding, interaction]
status: complete
updated: 2026-11-28
summary: "Affordance（可交互性/功能可供性）描述环境与物体在特定任务、场景和机器人本体条件下提供的可执行交互机会，贯穿感知（what/where）、推理（relevance）与动作引导（how）三段接口。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/grasp-pose-estimation.md
  - ../methods/vla.md
  - ../queries/robot-perception-stack-selection-loop.md
  - ../queries/contact-wrench-closed-loop.md
original:
  - ../../sources/papers/affordance_learning_embodied_ai_survey_preprints_2026100392.md
sources:
  - ../../sources/papers/affordance_learning_embodied_ai_survey_preprints_2026100392.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Affordance（功能可供性 / 可交互性）

## 一句话定义

**Affordance** 指环境或物体在当前任务上下文与智能体本体（embodiment）条件下所能支持的具体交互机会，把「画面/场景里有什么」转化为「能对哪里、以何种方式施动」的可接地中间表征。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| HOI | Human-Object Interaction | 人–物交互，常用于从人类动作视频提取交互区域与接触线索 |
| VLM | Vision-Language Model | 视觉–语言模型，用于开放词汇语义接地与 affordance 推理 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略，常把 affordance 作为显式条件或辅助监督 |
| 6-DoF | Six Degrees of Freedom | 六自由度位姿，用于描述抓取或接触后操作方向等执行规格 |
| KLD / IoU | Kullback–Leibler Divergence / Intersection over Union | 分别用于评估热图分布与掩码分割型 affordance 预测 |

## 为什么重要

- **弥合语义识别与物理执行之间的断层**：单纯识别物体类别（如「杯子」「抽屉」）不足以驱动控制；机器人还需要知道把手或杯耳在哪里、应以何种抓握方向与运动约束交互。
- **把高维视觉压缩为与动作强相关的低维接口**：将像素或点云提炼为区域掩码、热图、关键点、6-DoF 位姿或接触后轨迹，能提高策略在跨物体、跨场景泛化时的样本效率。
- **受本体与任务约束而非物体固有静态标签**：同一把剪刀在「递给他人」与「裁剪」任务中对应完全不同的交互部位与安全边界；同一个门把手对平行夹爪与灵巧手的可行动作也不同。

## 核心原理

依 [具身 AI Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md) 的形式化框架，affordance 学习可表达为映射 $F = f_\theta(o, l, c) = (a, r, \eta)$：给定观测 $o$、可选指令 $l$ 与上下文 $c$，输出可能的交互标签 $a$（what）、空间接地区域 $r$（where）及执行规格 $\eta$（how，如 6-DoF 位姿、动作得分、运动方向或接触后轨迹），再经分层执行 $u = g_{\mathrm{exec}}(F, C_{\mathrm{robot}})$ 或直接并入策略 $u = \pi_\phi(o, l, p, F)$。

```mermaid
flowchart LR
  O["多模态观测 + 任务指令 + 本体状态"] --> P["Affordance 感知: 定位可交互部位 / 接触点 / 3D 区域"]
  P --> R["Affordance 推理: 结合任务意图与本体可达性筛选"]
  R --> A["Affordance 引导动作: 原语 / 规划约束 / 策略条件或辅助监督"]
  A --> E["真机或仿真闭环执行与可达性验收"]
```

| 阶段 | 核心问题 | 典型表征形式 | 常见评估维度 |
|---|---|---|---|
| **Affordance 感知** | 哪里可以发生何种交互 | 2D 热图/掩码、关键点、3D 点云部件分割、方向向量 | KLD、SIM、NSS、IoU / mIoU、欧氏距离误差 |
| **Affordance 推理** | 当前任务与本体该选哪一处 | 关系图判断、VLM 空间指点、候选可执行排序 | 推理准确率、关系/可行性判断一致率 |
| **Affordance 引导动作** | 如何转化为机器人动作 | 原语参数、运动规划目标、Diffusion/VLA 显式条件或辅助损失 | 任务成功率、无碰撞可达率、真机纠错次数 |

## 工程实践

1. **按三段接口分别排查瓶颈**：部署失败时分别检查感知端是否定位准确、推理端是否结合末端几何/限位过滤不可行候选、动作端是否真正利用了 affordance 约束（对照「无 affordance / 预测 affordance / oracle affordance」三组消融）。
2. **明确区分 2D 热图精度与闭环执行成功率**：综述指出辅助 affordance 预测指标（KLD/IoU）提升并不自动保证下游操作成功率同步提高，评测时必须联报真机/仿真任务成功率、延迟与异常恢复。
3. **优先补齐 3D 度量接地与本体运动学校验**：2D 像素热图在遮挡与深度变化下容易漂移；接入机械臂执行前需结合深度/点云反投影、碰撞体及 IK 可达性校验（见 [抓取位姿估计](../methods/grasp-pose-estimation.md)）。

## 局限与风险

- **概念边界易与普通部件分割混淆**：不绑定「可执行动作」与「本体约束」的静态部件标签不是完整 affordance。
- **开环静态区域假设的局限**：软体/可变形物体与动态接触过程需要接触后运动学轨迹，而非单帧静态接触点。
- **综述归档边界**：本页核心形式化与三段接口归纳自 Preprints.org 未经同行评审的预印本综述及其配套索引；跨论文具体指标不可脱离原始实验协议直接横向排序。

## 关联页面

- [具身 AI Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md) — 感知 → 推理 → 动作三段式形式化主来源
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md) — 206 篇去重 affordance 论文逐条索引
- [机器人操作任务](../tasks/manipulation.md) — affordance 表征落地的核心任务域
- [抓取位姿估计](../methods/grasp-pose-estimation.md) — 接触点与 6-DoF 抓取候选的工程实现
- [VLA](../methods/vla.md) — 将 affordance 作为显式或隐式条件的端到端策略路线
- [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md) — 2D 区域到 3D 度量与策略消费的分层接口
- [接触力旋量闭环](../queries/contact-wrench-closed-loop.md) — 从可交互区域感知进入接触力估计、阻抗/力位控制与阶段执行的下游力控链

## 参考来源

- [From Passive Perception to Active Interaction 综述全文核对归档](../../sources/papers/affordance_learning_embodied_ai_survey_preprints_2026100392.md)
- [Awesome-Affordance-Learning 目录归档](../../sources/papers/affordance_awesome_catalog.md)
- [Awesome-Affordance-Learning 仓库归档](../../sources/repos/awesome-affordance-learning.md)

## 推荐继续阅读

- [Affordance Learning Survey 项目主页](https://jingliangli.com/affordance-survey/)
- [Awesome-Affordance-Learning GitHub](https://github.com/hq-King/Awesome-Affordance-Learning)
