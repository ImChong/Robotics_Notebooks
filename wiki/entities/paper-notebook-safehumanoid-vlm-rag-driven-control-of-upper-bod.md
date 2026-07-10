---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2511.23300"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_safehumanoid.md
summary: "SafeHumanoid 把「怎么调阻抗」这个低层控制问题，交给一个第一视角 VLM + RAG 检索库来回答：头部相机画面 → VLM 抽成结构化场景语义（任务、物体易碎性、是否有人、障碍等）→ 在 16 条经安全标准验证的模板库里做最近邻检索 → 取回每关节的刚度 Kp / 阻尼 Kd / 速度，下发给 50 Hz 的板载阻抗控制器。一旦画面里出现人手，机器人自动降刚度、升阻尼、减速，在不丢任务的前提下提升人机协作安全性。"
---

# SafeHumanoid

**SafeHumanoid: VLM-RAG-driven Control of Upper Body Impedance for Humanoid Robot** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

SafeHumanoid 把「怎么调阻抗」这个低层控制问题，交给一个第一视角 VLM + RAG 检索库来回答：头部相机画面 → VLM 抽成结构化场景语义（任务、物体易碎性、是否有人、障碍等）→ 在 16 条经安全标准验证的模板库里做最近邻检索 → 取回每关节的刚度 Kp / 阻尼 Kd / 速度，下发给 50 Hz 的板载阻抗控制器。一旦画面里出现人手，机器人自动降刚度、升阻尼、减速，在不丢任务的前提下提升人机协作安全性。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| VLM | Vision-Language Model | 视觉语言模型，本文用 Molmo-7B 把画面转成结构化语义 |
| RAG | Retrieval-Augmented Generation | 检索增强生成，这里是「查模板库取参数」 |
| Impedance | Impedance Control | 阻抗控制，用刚度 Kp / 阻尼 Kd 调节交互柔顺度 |
| IK | Inverse Kinematics | 逆运动学，把 6-DoF 目标位姿解成关节参考角 |
| FAISS | Facebook AI Similarity Search | 向量近邻检索库，做语义模板匹配 |
| HRI | Human-Robot Interaction | 人机交互 |
| ISO/TS 15066 | — | 协作机器人人机协作安全技术规范 |

## 为什么重要

- **人机协作安全**：提供一条「语义 → 阻抗」的可落地路径，把安全标准嵌进控制参数选择
- **VLM 接低层控制**：不让 LLM 直接吐扭矩，而是经检索库做中介，工程上更稳、更可控
- **模板库范式**：用少量人工验证模板 + 近邻检索，权衡了「可解释/合规」与「自动化」
- **延迟瓶颈**：也暴露了离板大模型 + 低频感知在动态 HRI 下的实时性天花板

## 解决什么问题

人形机器人和人**共享同一工作空间**做桌面操作（擦桌、递物、倒液体）时，安全的核心矛盾是：

1. **刚度高 = 精度好但危险**：硬碰到人手会产生大的接触力。 2. **刚度低 = 安全但任务做不好**：太软抓不稳、对不准。 3. 传统做法的阻抗参数是**固定 / 手调**的，无法随「现在面前是不是有人、物体易不易碎」自动变化。

## 核心机制

1. **语义驱动阻抗的闭环系统**：第一个把「VLM 看场景 → RAG 取参数 → 关节阻抗」串成在线闭环、并跑在真 G1 上的工作。
2. **安全标准入库**：模板库里的每条阻抗配置都按 ISO/TS 15066、ISO 13855 实测验证，把「合规」前移到检索库设计阶段。
3. **结构化检索而非自由生成**：用固定提示词 + FAISS 最近邻，换来确定、可复现、可兜底的参数输出，规避 LLM 端到端生成控制量的不稳定。
4. **一定泛化能力**：库里没有的物体（如 pin 销钉）也能被 VLM-RAG 归到一个合适的柔顺配置。

方法拆解（深读笔记小节）：第一视角感知（1–2 Hz）；VLM 抽语义（Molmo-7B）；RAG 两阶段检索；关节级阻抗执行（50 Hz，板载 Jetson Orin NX）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance.html> |
| arXiv | <https://arxiv.org/abs/2511.23300> |
| 发表 | 2025-11-28 (arXiv) |
| 源码 | 论文未公开代码 / 项目页 |
| 笔记阅读日期 | 2026-06-22 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_safehumanoid.md](../../sources/papers/humanoid_pnb_safehumanoid.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance.html>
- 论文：<https://arxiv.org/abs/2511.23300>

## 推荐继续阅读

- [机器人论文阅读笔记：SafeHumanoid](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance/SafeHumanoid__VLM-RAG-driven_Control_of_Upper_Body_Impedance.html)
