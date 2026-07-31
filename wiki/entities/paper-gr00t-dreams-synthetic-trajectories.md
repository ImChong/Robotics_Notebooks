---
type: entity
tags: ["paper", "synthetic-data", "world-model", "gr00t", "nvidia", "hmi-papers"]
status: complete
updated: 2026-07-31
venue: "HMI curated · 2025"
summary: "GR00T-Dreams（HMI P068）：NVIDIA 合成轨迹 blueprint：少真实遥操 post-train Cosmos → 语言生成视频 dreams → 筛选 → IDM 标动作 → 与真数据共训 VLA。"
related:
  - ./paper-hrl-stack-34-gr00t_n1.md
  - ./paper-notebook-dreamgen-unlocking-generalization-in-robot-learn.md
  - ./paper-hrl-stack-35-dreamdojo.md
  - ../concepts/world-action-models.md
  - ../entities/humanoid-motion-intelligence.md
sources:
  - ../../sources/papers/hmi_p068_gr00t-dreams-synthetic-trajectories.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# GR00T-Dreams（HMI P068）

**GR00T-Dreams**（*GR00T-Dreams: Synthetic Trajectory Generation for Humanoid Robot Learning*，2025）收录于具身智能研究室 [论文与项目总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md) **P068**，主分类为 **世界模型、VLA与Agent**。本页为本库独立详情节点（编译自策展解读与公开元数据，非原文镜像）。

## 一句话定义

NVIDIA 合成轨迹 blueprint：少真实遥操 post-train Cosmos → 语言生成视频 dreams → 筛选 → IDM 标动作 → 与真数据共训 VLA。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IDM | Inverse Dynamics Model | 从视频帧反推动作段 |
| VLA | Vision-Language-Action | 用神经轨迹后训练的策略 |
| WFM | World Foundation Model | Cosmos 等世界基座模型 |
| GR00T | Generalist Robot 00 Technology | NVIDIA 人形基座模型线 |

## 为什么重要

- 1. 先采集少量真实遥操轨迹，用来post-train Cosmos Predict-2，给世界模型注入目标机器人的外观、运动约束和环境。 2. 从一张初始图像和新语言指令生成大量2D视频“dreams”，扩展物体、背景和行为组合。 3. 用Cosmos Reason判断动作是否成功、场景是否合理，过滤明显失败或幻觉视频。 4. IDM读“前帧 + 后帧”，预测中间的3D动作段，把纯像素视频转成带动作标签的neural trajectory。 5. 将神经轨迹与真实数据共同训练或后训练VLA，再回到真机检验。
- 在 HMI 六条技术路线中属于 **世界模型、VLA与Agent**，补齐「总索引有条目、本库无下钻页」的缺口。
- 与相邻方法对照时，优先看问题设定与接口，而不是只记算法名。

## 核心信息

| 字段 | 内容 |
|------|------|
| HMI ID | P068 |
| 年份 | 2025 |
| 分组 | 世界模型、VLA与Agent |
| 开源状态 | blueprint/参考工作流；组件开源边界以 NVIDIA 博客与 Cosmos/GR00T 各仓为准 |
| 原文 | https://developer.nvidia.com/blog/enhance-robot-learning-with-synthetic-trajectory-data-generated-by-world-foundation-models/ |

## 核心原理

GR00T-Dreams是NVIDIA在2025年发布的一套blueprint/参考工作流，不是一篇单一模型论文。它的出发点是真机遥操太贵：先用少量某本体、某环境的真实示范教Cosmos这台机器人和任务的外观/运动，再用文字生成新场景和新动作视频，最后把合格视频反推成可训练的机器人动作。

### 流程直觉

```mermaid
flowchart LR
  A["问题 / 数据 / 观测"] --> B["GR00T-Dreams"]
  B --> C["控制 / 策略 / 数据产物"]
  C --> D["评测或真机闭环"]
```

模块边界与符号定义以原文为准；上图只固定阅读骨架。

## 工程实践

每条合成样本至少应保存初始图像、语言条件、生成视频、筛选分数、IDM动作块、目标本体schema和来源模型版本。只有视频而没有动作对齐，不能训练VLA；只有IDM动作而没有可追溯视频和筛选记录，也无法定位标签错误。合成数据与真实数据的batch比例、动作归一化和任务去重会直接影响后训练结果。

| 检查项 | 建议 |
|--------|------|
| 一手来源 | 回 arXiv / DOI / 项目页核对数值与声明 |
| 开源边界 | blueprint/参考工作流；组件开源边界以 NVIDIA 博客与 Cosmos/GR00T 各仓为准 |
| 本库定位 | 详情编译页；深入公式与实验表读原文 |

## 源码运行时序图

**不适用**（blueprint/参考工作流；组件开源边界以 NVIDIA 博客与 Cosmos/GR00T 各仓为准）。若后续官方发布可运行训练/推理入口，应补 `sources/repos/` 并更新本图。

## 实验与评测读法

- 把「仿真指标 / 真机证据 / 仅项目演示」分开记账。
- 对照同组相邻工作（见关联页面）时，对齐任务定义与观测接口，再比成功率。
- 综述类条目关注分类框架与缺口，不把引用列表当作选型排名。

## 结论

**GR00T-Dreams 应作为 HMI「世界模型、VLA与Agent」线上的独立知识节点阅读：先抓住其真正改变的问题接口，再决定是否进入复现或对比实验。**

- 核心贡献是问题表达或管线接口，而不只是单一网络结构名。
- 开源状态：blueprint/参考工作流；组件开源边界以 NVIDIA 博客与 Cosmos/GR00T 各仓为准。
- 与本库已有相邻页交叉阅读，避免重复造页。
- 数值、消融与许可以一手来源为准；本页是编译索引。
- 若官方后续补齐代码/数据，应回写 `sources/` 与本节开源字段。

## 局限与风险

- 视频看起来成功，不代表从两帧能唯一恢复动作。同一个末端位置变化可能对应不同关节轨迹、力量和接触历史，视频又可能隐藏遮挡和滑动。IDM如果在数据外反推出不可执行动作，会把视频幻觉转化为策略标签污染。因此真正可靠的管线需要不只用视觉语义筛选，还要检查动作限位、轨迹连续性、碰撞与真机小批量重放。
- 勿把 HMI 解读中的工程判断直接写成论文作者承诺。
- 经典控制论文与现代 RL/VLA 论文的「可复现」标准不同，选型时分开评估。

## 与其他工作对比

| 维度 | 本工作（GR00T-Dreams） | [DreamGen](paper-notebook-dreamgen-unlocking-generalization-in-robot-learn.md) | [GR00T N1](paper-hrl-stack-34-gr00t_n1.md) | [DreamDojo](paper-hrl-stack-35-dreamdojo.md) |
|------|------------------------|-------------------------------------------------------------------------------|--------------------------------------------|----------------------------------------------|
| 定位 | 合成轨迹数据生成 blueprint | 合成轨迹数据生成流水线 | 被训练的人形通才 VLA 策略 | robot world model（预测下一观测） |
| 世界模型用法 | post-train Cosmos 生成视频 dreams | 图像到视频生成模型产 neural trajectories | 数据消费方，不产数据 | 给定观测+动作预测未来 |
| 动作还原 | IDM 从前后帧反推 3D 动作段 | IDM 或 latent action model 恢复伪动作 | 直接由 flow-matching 动作头输出 | 不还原动作，做预测/规划底座 |
| 关系/取舍 | 与真数据共训 VLA；靠语义筛选，需查动作可执行性 | 同思路的相近工作，单任务数据泛化到 22 行为 | GR00T-Dreams 产的数据可喂给它 | 用途在规划/评估，既非策略也非数据生成 |

## 关联页面

- [HMI 论文覆盖导读](../queries/hmi-papers-coverage.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [paper-hrl-stack-34-gr00t_n1](./paper-hrl-stack-34-gr00t_n1.md)
- [paper-notebook-dreamgen-unlocking-generalization-in-robot-learn](./paper-notebook-dreamgen-unlocking-generalization-in-robot-learn.md)
- [paper-hrl-stack-35-dreamdojo](./paper-hrl-stack-35-dreamdojo.md)
- [world-action-models](../concepts/world-action-models.md)

## 参考来源

- [sources/papers/hmi_p068_gr00t-dreams-synthetic-trajectories.md](../../sources/papers/hmi_p068_gr00t-dreams-synthetic-trajectories.md)
- [sources/repos/humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)
- [HMI 论文总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md)

## 推荐继续阅读

- [项目/官方解读](https://developer.nvidia.com/blog/enhance-robot-learning-with-synthetic-trajectory-data-generated-by-world-foundation-models/)
- [HMI 逐篇解读 P068](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P068.md)
