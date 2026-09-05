# World Labs / Fei-Fei Li：世界模型功能分类（Renderer / Simulator / Planner）

> 来源归档（ingest）

- **标题：** A Functional Taxonomy of World Models
- **副标题：** Renderers, Simulators, Planners, and the Loop That Connects Them
- **类型：** blog
- **作者：** Fei-Fei Li 与 World Labs 团队
- **官方博客：** <https://www.worldlabs.ai/blog/taxonomy-of-world-models>
- **Substack 镜像：** <https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models>
- **日期：** 2026-06-03
- **入库日期：** 2026-09-05
- **一句话说明：** 用 POMDP 闭环把过载的「世界模型」拆成三类功能投影：Renderer 输出观测、Simulator 输出状态、Planner 输出动作；并主张仿真是枢纽，三类边界正在塌缩。

## 为什么值得保留

- 这是 2026 年对「world model」一词最有影响力的 **功能层消歧**：不按架构家族分，而按 **在 agent–环境环里输出什么** 分。
- 后续上海人工智能实验室视角文 [arXiv:2607.06401](https://arxiv.org/abs/2607.06401) 把它当作主要对话对象，并补上「内部表征怎么建」的第二轴。
- 把 [Marble](../../wiki/entities/marble-world-model.md) 明确放在 **Renderer ↔ Simulator** 交界（高斯 splat + collider），而不是具身像素 WM。

## 核心摘录（面向 wiki 编译）

### 1) 闭环先于分类

- 「世界模型」的技术含义来自 **POMDP**：动作改变潜状态，agent 只看见观测。
- **State** 是物理/机器人意义上的完整瞬时描述（物体、位姿、速度、属性），原则上完备、对内部 agent 不可直接见。
- 词源：Craik 1943 的 small-scale models；1980s–90s 进入神经网络。今天被叫作 world model 的系统，是这条环的 **不同投影**。

### 2) 三类功能

| 功能 | 输出 | 合同 | 文内例子 |
|------|------|------|----------|
| **Renderer** | 给人看的像素/观测 | 视觉保真，不必显式 3D | 文生视频；Genie 3；World Labs RTFM |
| **Simulator** | 几何/物理/动力学上可计算的 **状态** | 结构正确，供人与程序共同使用 | 解析物理引擎；可交互训练场 |
| **Planner** | 动作 | 观测 + 目标 → 下一步做什么 | VLA、MBRL、World Action Models |

- Renderer 的反面是 Planner：前者 `action → observation`，后者 `observation → action`。
- 三类 **不是** 互斥模块；同一套几何/物理/动力学知识原则上可投影到三种输出。

### 3) 为什么仿真是枢纽

- Renderer **商业最成熟**（视觉可信 ≠ 可训机器人 / 可设计建筑）。
- Planner **最诱人、最幼嫩**：近两年 demo 几乎都困在窄物体集、短时域实验室；厨房/仓/手术室级验证仍缺。
- Simulator 最少被公众讨论，却是二者的桥：语言是抽象、像素是投影，**几何/物理/动力学才是世界本身**。
- 数据不对称：互联网视频喂 Renderer；带物性/碰撞/质量的 3D 资产与机器人轨迹喂 Simulator/Planner，量级差几个数量级。
- 生成式仿真新风险：几何看起来对，却自交、错尺度 → 物理无意义。多物理（刚体/软体/流体/布料）仍极贵。

### 4) 边界塌缩与统一世界模型

- 预训练视频 Renderer 已开始作 **joint world-and-action** 骨干（Renderer → Planner）。
- Marble：同一模型出 splat（看）与 collider（给物理引擎）（Renderer → Simulator）。
- 逻辑终点：一个基础模型按下游消费者切换输出模态。
- 收束句（后被 2607.06401 几乎原句引用）：语言让机器 **谈论** 世界；世界模型让机器 **理解、想象、推理并与之交互**。

## 开源边界（步骤 2.5，2026-09-05）

- 本篇是概念文，**无训练代码 / 权重**。
- 官方入口是 World Labs 博客；Substack 为作者镜像，正文一致。
- 文内产品：[Marble](https://marble.worldlabs.ai/) 仍是 **部分开源**（生成闭源 SaaS+API；Spark / 交互示例开源），见 [worldlabs_marble_world_model.md](./worldlabs_marble_world_model.md)。
- 项目页核查：<https://www.worldlabs.ai/blog/taxonomy-of-world-models> 无 GitHub；结论 **确认未开源（概念文）**。

## 对 wiki 的映射

- 升格：[世界模型功能分类](../../wiki/concepts/functional-taxonomy-world-models.md)
- 对话论文：[世界模型定义与路线图](../../wiki/entities/paper-sa-2607-06401-a-definition-and-roadmap-for-world-models.md)（arXiv:2607.06401）
- 交叉：[生成式世界模型](../../wiki/methods/generative-world-models.md)、[WAM](../../wiki/concepts/world-action-models.md)、[World Labs](../../wiki/entities/world-labs.md)、[Marble](../../wiki/entities/marble-world-model.md)、[训练闭环三线 taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)
