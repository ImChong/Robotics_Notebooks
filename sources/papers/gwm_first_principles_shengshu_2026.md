# General World Models from First-Principles（生数 / 清华，2026）

> 来源归档

- **标题：** General World Models from First-Principles
- **类型：** paper / technical report / manuscript
- **作者：** Jun Zhu, Hengkai Tan, Jintao Zhang, Min Zhao, Fan Bao, Bo Zhang 等（生数科技 / 清华大学）
- **机构：** 生数科技（Shengshu Technology）；清华大学
- **出处：** 2026 手稿 / WRC 2026 主题演讲配套（**无 arXiv**）
- **战略页：** https://www.shengshu.com/en/general-world-model/（入库日重定向至首页；以 WRC 发布稿与 [Motus2](https://motus-robotics.github.io/motus2/) 引用为准）
- **入库日期：** 2026-09-10
- **一句话说明：** 从第一性原理定义 GWM 为理解–想象–行动闭环，给出 L1–L5 能力路线图与 D1–D5 数据金字塔，并以 MoT 统一理解/生成/行动专家。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-gwm-first-principles.md`](../../wiki/entities/paper-gwm-first-principles.md)

---

## 核心主张（编译自 WRC 2026 发布稿与 Motus2 引用）

### GWM 定义

通用世界模型 **不是** 单独的生成器、模拟器或策略，而是 **理解（Understanding）→ 想象（Imagination）→ 行动（Action）→ 反馈** 的闭环系统；预测是连接感知、规划与行动的桥梁，而非孤立能力。

### L1–L5 路线图

| 级别 | 名称 | 能力 | 生数对应产品（文内 / 发布稿） |
|------|------|------|------------------------------|
| L1 | World Generation | 生成看起来合理的世界 | Vidu Q3 |
| L2 | Interactive World | 用户输入改变后续生成 | Vidu S1 |
| L3 | Actionable World | 动作进入世界、反馈修正控制 | Motus / **Motubrain**（WAM） |
| L4 | Autonomous World Agent | 仅给高层目标，自主分解与探索 | **尚无系统实现** |
| L5 | World Orchestrator | 多智能体/多机器人协同组织 | **尚无系统实现** |

### D1–D5 数据金字塔（报告口径）

| 层 | 内容 | 作用 |
|----|------|------|
| D1 | 互联网视频 | 世界广度：物体如何动、任务如何展开 |
| D2 | 教学 / 解说视频 | 任务结构与人如何完成 |
| D3 | 第一视角人类视频 | 从行动者视角观察环境 |
| D4 | 带动作记录的人类示范 | 连接「做了什么」与「世界如何变」 |
| D5 | 真实机器人轨迹 | 本体校准：可达空间、夹爪力、真机对齐 |

**注意：** 与 [PKU Data Pyramid（arXiv:2607.24744）](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md) **层级编号相似、分层逻辑不同**——勿混为同一框架。

### 架构与 scaling

- **MoT（Mixture-of-Transformers）**：理解 / 生成 / 行动专家共享注意力，维持同一「现在」。
- **Scaling 三维：** 知识覆盖扩大；视觉–语言–动作共享持续更新状态；推理速度跟上物理世界。

## 开源状态（步骤 2.5，2026-09-10）

| 资源 | 状态 |
|------|------|
| 手稿 / 演讲 PDF | **部分发布** — WRC 2026 主题演讲与新闻稿；无独立 arXiv |
| 战略落地页 | shengshu.com/general-world-model 入库日 **重定向** |
| 可运行训练代码 | **无** — L3 实例见 [Motus](./../../wiki/entities/paper-sa-2512-13030-motus-a-unified-latent-action-world-model.md)（已开源）、[Motubrain](./../../wiki/entities/paper-motubrain.md)（仓占位） |
| L4–L5 | **未实现** |

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-gwm-first-principles.md`](../../wiki/entities/paper-gwm-first-principles.md)
- 功能分类对照：[`wiki/concepts/functional-taxonomy-world-models.md`](../../wiki/concepts/functional-taxonomy-world-models.md)
- L3 产品实例：[Motubrain](../../wiki/entities/paper-motubrain.md)、[Motus2](../../wiki/entities/paper-motus2.md)
- 公众号盘点：[`sources/blogs/wechat_embodied_station_gwm_closed_loop_2026-09-10.md`](../blogs/wechat_embodied_station_gwm_closed_loop_2026-09-10.md)
