# TOSS Framework（arXiv:2608.21083）

> 来源归档（ingest）

- **标题：** Teaching is a Process: The TOSS Framework for Modeling Human Teaching Decisions in Human-Interactive Robot Learning
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2608.21083>
  - <https://osf.io/fumd8/?view_only=9cec60dccbd446f08bd818d0b3612705>
- **机构：** 莱顿大学（Leiden University）；阿姆斯特丹自由大学（VU Amsterdam）
- **入库日期：** 2026-08-25
- **一句话说明：** 34 名参与者观察两类 RL 场景共 204 条直觉教学反应，归纳 Triggers/Objectives/Signals/Strategies 四维网络，提出 TOSS 框架并开放 OSF 数据集。

## 核心摘录（MVP）

### 1) 自下而上探索研究设计

- **摘录要点：** 非交互观察范式：参与者观看 tabular Q-learning 导航与 DDPG 操作两类机器人学习视频（早/中/晚三阶段），回答「你会如何帮助机器人学得更好」以捕获无策略补偿的直觉教学逻辑。
- **对 wiki 的映射：**
  - [TOSS Framework](../../wiki/entities/paper-toss-framework.md) — 实验设计。
  - [reinforcement-learning](../../wiki/methods/reinforcement-learning.md) — HIRL 语境。

### 2) TOSS 四维结构

- **摘录要点：** **Triggers**（情境催化剂）→ **Objectives**（主观教学目标）→ **Signals**（沟通/反馈行为）→ **Strategies**（高层教学治理）；教师自然切换教练、工程师、设计者等角色。
- **对 wiki 的映射：**
  - [TOSS Framework](../../wiki/entities/paper-toss-framework.md) — 框架定义。

### 3) 开放数据与理论用途

- **摘录要点：** OSF 公开问卷、编码方案、刺激材料与数据语料；框架可用于建模 realistic oracle、解释人类教学决策、重设计人机教学设置。
- **对 wiki 的映射：**
  - [toss-framework-osf](../../sources/sites/toss-framework-osf.md) — 数据入口。

### 4) 开源状态（截至 2026-08-25）

- **摘录要点：** **已开源** — OSF 数据集与实验材料可公开获取（view-only 链见项目页）。
- **对 wiki 的映射：**
  - [TOSS Framework](../../wiki/entities/paper-toss-framework.md) — 工程实践表。

## 当前提炼状态

- [x] arXiv + OSF 已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-toss-framework.md` 新建
