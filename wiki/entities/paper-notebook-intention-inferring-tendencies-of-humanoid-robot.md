---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2508.04931"
related:
  - ../overview/paper-notebook-category-08-navigation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_intention.md
summary: "INTENTION 想让机器人不再依赖精确物理模型 + 预编排动作序列，而是像人一样凭直觉与环境交互：用 Grounded VLM 做场景推理，用 Memory Graph（记忆图） 把过去任务交互沉淀成\"经验\"，再用 Intuitive Perceptor（直觉感知器） 从图像里抽取物理关系与可供性（affordance），三者合起来在新场景也能推断出合适的交互行为，而不需要反复给指令。"
---

# INTENTION

**INTENTION: Inferring Tendencies of Humanoid Robot Motion Through Interactive Intuition and Grounded VLM** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：08_Navigation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

INTENTION 想让机器人不再依赖精确物理模型 + 预编排动作序列，而是像人一样凭直觉与环境交互：用 Grounded VLM 做场景推理，用 Memory Graph（记忆图） 把过去任务交互沉淀成"经验"，再用 Intuitive Perceptor（直觉感知器） 从图像里抽取物理关系与可供性（affordance），三者合起来在新场景也能推断出合适的交互行为，而不需要反复给指令。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| VLM | Vision-Language Model | 视觉 - 语言基础模型 |
| Grounded VLM | —— | 把 VLM 的语义输出**对齐 / 落地**到具体物体、空间与可执行行为 |
| Affordance | —— | 可供性：物体"能被怎么用"的属性（可抓、可推、可拉…） |
| Memory Graph | —— | 记忆图：以图结构存储历史交互场景与决策 |

## 为什么重要

- **直觉式交互范式**：把"人凭直觉动手"建模成 VLM + 记忆 + 可供性的组合，是模型驱动控制之外的一条思路
- **记忆驱动决策**：Memory Graph 把"经验"显式化，为长程、多任务的人形自主操作提供可累积的知识载体
- **VLM 落地**：展示了 Grounded VLM 如何从"会描述"走到"会动手"，对具身 VLA / 导航操作类工作有借鉴价值

## 解决什么问题

传统操作控制 / 规划路线高度依赖两样东西：

1. **精确的物理模型**——一旦实际物体的质量、摩擦、形状与模型有出入，控制就容易失败； 2. **预定义的动作序列**——在结构化环境里有效，但**换个新任务 / 新物体就难以泛化**。

## 核心机制

1. **提出 INTENTION 框架**：用「交互直觉」替代「精确模型 + 预定义序列」，让人形机器人在新场景里自主推断交互行为。
2. **Memory Graph**：以图结构沉淀历史交互经验，提供"类人"的任务记忆与决策类比能力。
3. **Intuitive Perceptor**：显式从视觉场景抽取物理关系与可供性，把人类的物理直觉接口化。
4. **Grounded VLM 落地**：把 VLM 的语义推理对齐到具体物体与可执行行为，打通"看懂场景 → 动手"的链路。
5. **去指令依赖**：核心卖点是**无需重复指令**即可泛化到未见场景，降低对人工编排的依赖。

方法拆解（深读笔记小节）：Grounded VLM —— 场景语义推理底座；Intuitive Perceptor（直觉感知器）—— 从图像抽物理关系与可供性；Memory Graph（记忆图）—— 把历史交互沉淀成经验；三者协同。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 08_Navigation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/INTENTION__Inferring_Tendencies_of_Humanoid_Robot_Motion_Through_Interactive_Int/INTENTION__Inferring_Tendencies_of_Humanoid_Robot_Motion_Through_Interactive_Int.html> |
| arXiv | <https://arxiv.org/abs/2508.04931> |
| 发表 | 2025-08-06 (arXiv) |
| 源码 | 截至当前未见公开发布（论文未给出 GitHub / 项目页链接） |
| 笔记阅读日期 | 2026-06-24 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-08-navigation](../overview/paper-notebook-category-08-navigation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_intention.md](../../sources/papers/humanoid_pnb_intention.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/INTENTION__Inferring_Tendencies_of_Humanoid_Robot_Motion_Through_Interactive_Int/INTENTION__Inferring_Tendencies_of_Humanoid_Robot_Motion_Through_Interactive_Int.html>
- 论文：<https://arxiv.org/abs/2508.04931>

## 推荐继续阅读

- [机器人论文阅读笔记：INTENTION](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/INTENTION__Inferring_Tendencies_of_Humanoid_Robot_Motion_Through_Interactive_Int/INTENTION__Inferring_Tendencies_of_Humanoid_Robot_Motion_Through_Interactive_Int.html)
