# 亮源新创发布机器人全身韧性智能技术 Light REACT

> 来源归档（blog / 微信公众号）

- **标题：** 亮源新创发布机器人全身韧性智能技术 Light REACT
- **类型：** blog
- **作者：** 亮源新创（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/Xfps8-XAv3u1S--EpS5Ezw
- **发表日期：** 2026-09-09
- **入库日期：** 2026-09-09
- **抓取方式：** `urllib` + HTML 解析（`js_content`）；`--no-images`
- **一句话说明：** 亮源新创「规模化部署」范式首个成果：把故障/扰动下的全身适应建模为 **具身 In-Context Learning**——以近期全身交互历史为上下文、**权重不变** 地调整步态/爬行/摔倒恢复；REACT = **REsilient humAnoid ConTrol**。

## 核心摘录（归纳，非全文）

### 定位：三段范式之「规模化部署」

- 继 **2026-09-01** [LightNav-0](../papers/lightnav0_arxiv_2608_30935.md)（规模化对齐）后，Light REACT 面向 **真实运行中的全身控制挑战**。
- 亮源新创三段范式：**规模化预训练 → 规模化对齐 → 规模化部署**；本篇为部署段首个公开成果。

### 问题：关节失效与扰动下的持续运行

- 规模化部署要求运行条件变化后仍能 **自主适应**；部分关节失效时，同一动作指令可能产生不同运动结果。
- 若异常后仍依赖现场人员或 **重新训练**，持续运行受限。
- Light REACT 将 **故障自适应行走、爬行与摔倒恢复** 整合进 **同一控制策略**：
  - 部分关节失效 → 协调可用关节、调整步态与发力；
  - 无法站立 → 调用手臂等部位 **转入爬行**；
  - 外部扰动摔倒 → 根据当前运动能力尝试恢复。
- 文内 **仿真测试**：整条腿失去供电时，可调整全身动作继续行走或爬行，**无需人工切换模式**。

### 机制：具身 In-Context Learning / 全身上下文学习

- 把故障自适应建模为 **具身 In-Context Learning（ICL）**：
  - **输入上下文**：近期全身交互历史（运动反馈：预期动作未完成、实际运动与原先不同）。
  - **无需** 故障部位/程度标签；**无需** 人工切换控制模式。
  - **模型参数保持不变**；交互上下文持续更新 → 动作随之调整。
- 亮源新创命名：**Whole-Body Context Learning（全身上下文学习）**——覆盖行走步幅/发力调整，以及行走、起身、爬行之间的行为重组。
- **REACT** = **REsilient humAnoid ConTrol**（人形机器人韧性控制）。

### 训练与部署

| 阶段 | 做法 |
|------|------|
| **训练** | 仿真合成大量 **全身交互上下文数据**，覆盖不同故障条件下的行走、爬行、摔倒恢复；训练 **Transformer 策略**，学会利用连续交互历史推断当前运动能力并调整全身动作 |
| **部署** | 无故障标签、无在线权重更新；以近期全身交互历史为上下文 **自主适应** |

### 规模化叙事

- 每台机器经历不同扰动/故障/运行状态；同一模型应能利用各自交互经验应对差异。
- 持续运行经验回流后续训练与模型迭代（数据飞轮）。

## 开源核查（入库日 2026-09-09）

| 资产 | 状态 |
|------|------|
| 官方项目页 | **未列**（[lightorigins.com](https://www.lightorigins.com/) 博客区截至入库日仅有 LightNav-0、LightParkour） |
| 论文 / arXiv | **未列** |
| 代码 / 权重 | **未列** GitHub 或 Hugging Face 链接 |

> 以微信公众号发布与官网可访问页面为准；后续若上线项目页或开源仓，应在 `sources/sites/light-react.md` 与 wiki 实体页同步更新。

## 对 wiki 的映射

- **实体页（新建）：** [light-react](../../wiki/entities/light-react.md)
- **项目页归档（新建）：** [light-react](../sites/light-react.md)
- **概念交叉：** [robot-in-context-learning](../../wiki/concepts/robot-in-context-learning.md)（全身上下文适应 vs 操作臂 ICL）
- **机构关联：** [paper-lightnav-0](../../wiki/entities/paper-lightnav-0.md)、[paper-light-loco-parkour](../../wiki/entities/paper-light-loco-parkour.md)（同机构不同能力轴）
