---
type: concept
tags: [dataset, scaling, cross-embodiment, manipulation, community]
status: complete
updated: 2026-11-15
related:
  - ../entities/paper-open-x-embodiment.md
  - ../queries/contact-wrench-closed-loop.md
  - ./foundation-policy.md
  - ./embodied-scaling-laws.md
  - ../entities/paper-octo.md
  - ../methods/vla.md
  - ../entities/paper-topreward.md
  - ../entities/paper-data-pyramid-embodied-manipulation.md
  - ../entities/paper-ucag-p.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
  - ../../sources/papers/topreward_arxiv_2602_19313.md
  - ../../sources/papers/data_pyramid_embodied_manipulation_arxiv_2607_24744.md
summary: "Open X-Embodiment（OXE）联合多机构把异构机器人演示数据规范化并开源，支撑跨本体规模化学习与通用策略预训练。"
original:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
---

# Open X-Embodiment（OXE）

## 一句话定义

**Open X-Embodiment**：面向机器人模仿学习的大规模跨机构、跨硬件形态数据集与基准管线，把多种机器人的演示统一到可比格式上，用于训练与评测「通用操作策略」。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OXE | Open X-Embodiment | 跨形态机器人的大规模操作数据集 |
| RT | Robotics Transformer | 早期通用操作策略系列，常与 OXE 数据叙事并列 |
| VLA | Vision-Language-Action | 在 OXE 等混合数据上预训练的通用策略形态 |

## 为什么重要

它为 [Embodied Scaling Laws](./embodied-scaling-laws.md) 与 [Foundation Policy](./foundation-policy.md) 叙事提供了可公开核验的数据轴：在同一大混合上预训练的策略（如 [Octo](../entities/paper-octo.md)）成为后续微调与对比实验的默认起点。

## 核心原理

- **跨具身统一 Schema**：通过 RLDS/TFDS 将 60+ 数据集、22 类本体的观测与动作字段映射到统一存储 schema，降低多源数据联合加载的工程门槛。
- **跨本体表征迁移**：在统一视觉与末端动作接口上联合训练（如 RT-1-X / RT-2-X / Octo），让策略从多本体经验中提取共享的几何与语义先验。

## 工程实践

- **按需筛选子集而非盲目全量混合**：RT-X 实际仅选取 9 种本体进入同一模型；工程接入时应先核对观测视角、动作坐标系与控制频率。
- **保留本体与数据源元数据**：联合训练或迁移评测时必须保留 `embodiment` 与任务描述元数据，避免不可比轨迹混入同一批次。

## 局限

- **统一的是存储 schema，不是动力学**：不同机器人的自由度、连杆长度与接触动力学差异并未被格式消除，不支持零样本控制任意未见本体。

## 关联页面

- [机器人本体（Embodiment）](./embodiment.md) — 区分存储格式统一与身体、观测、动作及控制契约的一致性
- [具身智能数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md) — 按形态分组的数据集入口与节点映射
- [Query：接触力旋量闭环知识链](../queries/contact-wrench-closed-loop.md) — 跨本体操作数据训练的策略，其接触执行环节由本链托底
- [Octo Model](../entities/paper-octo.md)
- [Foundation Policy](./foundation-policy.md)
- [TOPReward](../entities/paper-topreward.md) — 在 OXE 上评测零样本进度奖励（Mean VOC；项目页 / v1 口径 39 个数据集，v2 论文过滤为 33 个）
- [具身数据金字塔综述](../entities/paper-data-pyramid-embodied-manipulation.md) — 把 OXE 定位为五层金字塔真机层的聚合代表，并给出跨层数据选型坐标系
- [UCAG-P](../entities/paper-ucag-p.md) — 在异构机器人+人手小时上用 **相机系锚点** 作共享动作接口（非 OXE 格式本身）

## 参考来源

- Padalkar et al., *Open X-Embodiment: Robotic Learning at Scale*, https://arxiv.org/abs/2310.08864
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
- [sources/papers/topreward_arxiv_2602_19313.md](../../sources/papers/topreward_arxiv_2602_19313.md) — OXE 进度估计评测轴（含 v2 全文核对）
- [sources/papers/ucag_p_arxiv_2608_26058.md](../../sources/papers/ucag_p_arxiv_2608_26058.md) — 相机系锚点作为异构源上的共享动作接口

## 推荐继续阅读

- [OXE / RT-X 实体详情页](../entities/paper-open-x-embodiment.md)
- [Open X-Embodiment 项目主页](https://robotics-transformer-x.github.io/)
