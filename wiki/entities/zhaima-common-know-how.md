---
type: entity
tags:
  - knowledge-note
  - vla
  - robot-learning
  - secondary-source
  - zhaima
status: complete
updated: 2026-10-22
related:
  - ../methods/vla.md
  - ../methods/diffusion-policy.md
  - ../methods/action-chunking.md
  - ./cosmos-3.md
  - ./isaac-gr00t.md
  - ../methods/pi07-policy.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/zhaima_embodied_know_how_feishu.md
summary: "宅码《Common Know-How》（飞书文档转载）：汇集 VLA 数据集、FK/IK、RL、流匹配与扩散、π0 动作表示、Cosmos 视频 tokenization、RTC 与记忆等知识点的二手笔记；本页为独立信息节点，结论以原文与一手论文为准。"
---

# 宅码 Common Know-How（二手知识点汇编）

## 一句话定义

**这是一份由「宅码」公众号转载到飞书的 VLA 与机器人学习知识点汇编，适合作为入门索引，但它是二手笔记，结论需回到一手论文核对。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉语言动作模型，文档的主要对象 |
| FK / IK | Forward / Inverse Kinematics | 正向 / 反向运动学，关节角与末端位姿之间的转换 |
| RL | Reinforcement Learning | 强化学习，文中含奖励与 Q 值的递推示例 |
| RTC | Real-Time Chunking | 实时动作分块，文中的推理时序融合方法 |
| MEM | Multi-scale Episodic Memory | π0.6 相关的多尺度记忆机制（文中名称） |

## 为什么重要

- 文档把数据集、运动学、强化学习、生成模型与 VLA 的动作表示放在同一份笔记中，适合快速建立术语地图。
- 它来自转载与二次整理，不是论文原文；作为索引使用比作为依据更合适。

## 核心结构

文档按主题排列，主要包括：

1. **数据集：** 常用机器人数据集的规模与特点（如 OXE、DROID、LIBERO）。
2. **运动学与控制链路：** FK / IK 的直观例子与 VLA 到低层控制器的流程。
3. **强化学习：** 奖励、Q 值递推与在线人机协作 RL（HIL-SERL）。
4. **生成模型：** 流匹配、DiT、I-JEPA、diffusion policy、CVAE 的对照。
5. **VLA 与动作表示：** RT-2、OpenVLA、π0 的自回归与扩散两路，以及 DCT / BPE 动作 tokenization。
6. **视频世界模型：** Cosmos 的扩散与离散两条路线。
7. **推理与训练方法：** RTC、SmolVLA、Wall-OSS、PLD，以及 π0.6 的记忆与 RECAP。

## 与其他页面的关系

- [VLA 方法](../methods/vla.md)、[Diffusion Policy](../methods/diffusion-policy.md)、[Action Chunking](../methods/action-chunking.md)：文档相关主题的 wiki 对应页。
- [Cosmos 3](./cosmos-3.md)：文档中 Cosmos 扩散 / 离散两路的一手材料。
- [Isaac GR00T](./isaac-gr00t.md)、[π0.7 策略](../methods/pi07-policy.md)、[Generative World Models](../methods/generative-world-models.md)：文档涉及的相关路线。

## 工程实践

- 作为入门时的术语索引：遇到不熟悉的方法名，先查对应 wiki 页，再回到原论文。
- 文中的具体数字与结论（例如数据集规模、动作表示的优劣比较）应以原论文为准。

## 局限与风险

- **二手来源：** 文档为公众号转载，原作者与原始论文的对应关系未逐条标注。
- **结论的适用范围：** 文中有将单篇论文结论泛化为一般规律的表述（如动作表示的优劣排序），需要结合原实验条件看。
- **版权：** 原文为转载，本页只保留主题与链接，不转存全文。
- **覆盖未核实：** 本页的主题索引是关键词级抽取，并未逐段通读。

## 参考来源

- [宅码 Common Know-How 飞书归档](../../sources/blogs/zhaima_embodied_know_how_feishu.md)

## 推荐继续阅读

- [飞书原文](https://my.feishu.cn/docx/KXoqdn7FooTIZlxCHX8cQJnknVd)
- [VLA 方法](../methods/vla.md)
