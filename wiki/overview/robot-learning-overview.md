---
type: overview
updated: 2026-04-20
summary: "Robot Learning Overview 是机器人学习知识库入口页，按控制层、学习层与系统层组织主干主题，并给出阅读顺序与问题驱动导航。"
sources:
  - ../../sources/papers/survey_papers.md
---

# Robot Learning Overview

**机器人学习**：让机器人通过数据学会完成复杂任务的方法集合，核心是把”如何做”从人工编程转向从经验中学习。

## 为什么重要

传统机器人依赖精确建模和手工控制，缺点：

- 对复杂环境建模困难
- 手工调参成本高
- 泛化能力差

机器人学习的优势：

- 从数据中自动提取有效策略
- 能处理高维状态/动作空间
- 在仿真中训练，迁移到真实机器人

## 核心组成

### 1. 强化学习（RL）
从环境交互中学习最优策略。核心是 reward signal。

### 2. 模仿学习（IL）
从专家演示中学习策略。核心是 demonstration。

### 3. 预测模型
学习环境动态模型，用于 planning 或 model-based RL。

### 4. 感知-动作联合学习
端到端学习直接从 raw sensor 到 action 的映射。

## 三层架构：从策略到系统

如果把机器人学习只理解成“用一个神经网络输出动作”，就很容易在真实机器人上踩坑。更实用的理解方式是把它拆成三层：

### 1. 任务与策略层

这一层回答“机器人要做什么、策略学什么”：

- 强化学习：适合从交互中学习长期回报
- 模仿学习：适合从人类示范中快速学到可用行为
- VLA / Foundation Policy：把语言、视觉和动作统一起来，适合多任务操作

这层的难点是数据、奖励和泛化，而不是高频执行本身。

### 2. 控制与执行层

这一层回答“高层策略给出的目标，怎么安全稳定地变成机器人动作”：

- 对人形机器人，常见执行层包括 [Whole-Body Control](../concepts/whole-body-control.md)、[TSID](../concepts/tsid.md)、[Model Predictive Control](../methods/model-predictive-control.md)
- 对操作任务，常见执行层包括 [Impedance Control](../concepts/impedance-control.md)、力位混合控制和任务空间控制
- 对带安全约束的系统，还会加入 [Control Barrier Function](../concepts/control-barrier-function.md) 或 safety filter

很多“学习失败”其实发生在这里：策略输出没问题，但执行层接不住，或者真机频率、延迟和约束根本不匹配。

### 3. 系统与部署层

这一层回答“训练好的东西如何真实跑起来”：

- 感知和状态估计是否稳定
- 训练分布和真机分布是否一致
- 控制频率、通信延迟、动作限幅是否处理妥当
- 数据能否从失败部署里回流继续改进

这也是 [Sim2Real](../concepts/sim2real.md)、[Sensor Fusion](../concepts/sensor-fusion.md)、[System Identification](../concepts/system-identification.md) 变得关键的原因。

## 五大主题导航

当前知识库里的机器人学习主线，基本可以归成五个主题：

### 1. Locomotion

研究机器人如何稳定移动，包括平地行走、地形适应、扰动恢复和全身平衡。它通常和 RL、MPC、WBC 紧密相关，是人形机器人最核心的能力底座。

入口页：

- [Locomotion](../tasks/locomotion.md)
- [Terrain Adaptation](../concepts/terrain-adaptation.md)
- [Query：人形机器人运动控制 Know-How](../queries/humanoid-motion-control-know-how.md)

### 2. Manipulation

研究机器人如何抓取、装配、双臂协同和执行接触丰富任务。它是模仿学习、遥操作、VLA 和 Foundation Policy 最集中的应用场景。

入口页：

- [Manipulation](../tasks/manipulation.md)
- [Bimanual Manipulation](../tasks/bimanual-manipulation.md)
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)

### 3. Learning Methods

研究机器人策略本身如何从数据中学出来，包括 RL、IL、Behavior Cloning、Diffusion Policy、VLA 等。它们不是互斥路线，而是经常组合使用：例如 BC 预训练 + RL 微调，或 VLA 负责语义层、传统控制负责执行层。

入口页：

- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [Behavior Cloning](../methods/behavior-cloning.md)
- [VLA](../methods/vla.md)

### 4. Safe / Structured Control

研究如何在有物理约束和安全约束的情况下，让系统既稳定又可部署。这里会自然连到 Lyapunov、CLF、CBF、WBC 与 MPC。

入口页：

- [Whole-Body Control](../concepts/whole-body-control.md)
- [Control Lyapunov Function](../formalizations/control-lyapunov-function.md)
- [Control Barrier Function](../concepts/control-barrier-function.md)
- [Query：CLF 与 CBF 在 WBC/MPC 中的联合使用](../queries/clf-cbf-in-wbc.md)

### 5. Sim2Real 与系统落地

研究如何把仿真中的学习结果迁移到真实机器人上，包括域随机化、系统辨识、状态估计、调试和部署流程。对于真实机器人项目，这一主题往往决定一个策略能不能真正落地。

入口页：

- [Sim2Real](../concepts/sim2real.md)
- [Domain Randomization](../concepts/domain-randomization.md)
- [Query：Sim2Real 真机部署清单](../queries/sim2real-deployment-checklist.md)
- [Query：RL 策略真机调试 Playbook](../queries/robot-policy-debug-playbook.md)

## 当前图谱里的几个 community 应该怎么读

图谱社区并不是严格学科分类，而更像“在链接上天然靠得近的一组页面”。当前最值得关注的几组主干通常是：

- **Locomotion 社区**：从 LIP/ZMP、Centroidal Dynamics、MPC、WBC 一路连到 locomotion 与 sim2real，适合走传统运动控制主线
- **WBC / Safe Control 社区**：围绕 TSID、HQP、CLF、CBF、接触约束和安全过滤，适合想做可证明稳定/安全控制的人
- **IL / Manipulation 社区**：围绕遥操作、行为克隆、Diffusion Policy、VLA、操作与双臂任务，适合想做操作与多模态策略的人
- **DL / Formalization 社区**：围绕 [深度学习基础](../concepts/deep-learning-foundations.md)、Lyapunov、LQR、MDP、Bellman 等基础形式化页面，适合补理论地基

这些社区之间并不是割裂的。一个成熟的机器人系统往往同时跨过多个社区：例如 VLA 提供高层语义目标，WBC 负责高频执行，Sim2Real 负责部署闭环。

## 如果你准备系统学习，推荐怎么进入

### 路线 A：先把控制主干学通

如果你想做人形运动控制、MPC、WBC、状态估计，建议从 [Humanoid Control Roadmap](../roadmaps/humanoid-control-roadmap.md) 或 [Locomotion](../tasks/locomotion.md) 一路往下读。这样能先建立“机器人为什么需要这些结构”的系统直觉。

### 路线 B：从数据驱动方法切入

如果你已经在做操作或模仿学习，建议从 [Manipulation](../tasks/manipulation.md) → [Behavior Cloning](../methods/behavior-cloning.md) → [Diffusion Policy](../methods/diffusion-policy.md) → [VLA](../methods/vla.md) 这条线进入，再回头补执行层和 sim2real。

### 路线 C：以 query 页做问题驱动阅读

如果你不是线性系统学习，而是带着明确问题来，例如：

- “做 sim2real 部署先查什么？”
- “VLA 怎么和低层控制器接？”
- “接触丰富操作常见失败怎么修？”

那就直接先读对应 query 页，再顺着 `## 关联页面` 回到概念页和方法页。这也是知识库目前最符合工程问题驱动的一种读法。

## 学习建议

- **先建立主干，再看花样。** 没搞清楚 WBC、MPC、状态估计，就很容易把 RL/VLA 当黑盒 magic。
- **把 query 页当“高价值经验回写”。** 它们不是附录，而是从具体问题反推知识结构的最好入口。
- **注意任务层和执行层的边界。** 很多模型文章写的是“高层表现”，但真实落地成败取决于执行层与部署层。
- **尽量沿着相关页面跳转阅读。** 这个仓库最核心的价值不是某一页写得多，而是页面之间已经形成了交叉引用的图结构。

## 和其他领域的关系

- **控制理论**：提供稳定性、收敛性理论支撑
- **优化**：RL 训练本质是优化问题
- **计算机视觉**：感知模块的基础
- **运动控制**：具体执行层面的基础

## 知识库维护方法论

本知识库采用 Karpathy LLM Wiki 模式构建：LLM 作为维护者，人类作为 curator，通过 Ingest / Query / Lint 三类操作持续积累知识。

- [LLM Wiki 方法论（Karpathy）](../references/llm-wiki-karpathy.md) — 知识库构建模式来源
- [Humanoid Control Roadmap](../roadmaps/humanoid-control-roadmap.md) — 人形机器人控制的学习路线

## 参考来源

- Sutton & Barto, *Reinforcement Learning: An Introduction* — RL 标准教材
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com) — 实践入门

## 关联页面

- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [Locomotion](../tasks/locomotion.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [LLM Wiki 方法论（Karpathy）](../references/llm-wiki-karpathy.md) — 本知识库的构建方法论来源
- [Query：人形机器人运动控制 Know-How](../queries/humanoid-motion-control-know-how.md) — 实战经验结构化摘要，快速入门推荐

## 推荐继续阅读

- Sutton & Barto, *Reinforcement Learning: An Introduction*
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com)
