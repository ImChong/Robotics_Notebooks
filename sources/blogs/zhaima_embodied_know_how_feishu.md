# 宅码：具身智能 Common Know-How（飞书文档）

## 元数据

- **标题：** Common Know-How（微信公众号「宅码」转载的飞书文档）
- **链接：** https://my.feishu.cn/docx/KXoqdn7FooTIZlxCHX8cQJnknVd
- **类型：** blog（二手汇编，属转载）
- **抓取方式：** `scripts/fetch_feishu_doc.py`，832 块，约 6.8 万字符
- **入库日期：** 2026-10-22
- **版权：** 原文为转载，本归档只保留链接与提炼线索，不整篇转存。

## 与本次 ingest 条目的关系

- 抓取文本中**未检索到** EmbodyHOI、Action Influence Maximization、ERR、Agensh、StairMaster、EVO-WAM 的标题或编号，因此本文档不作为上述论文的来源。
- 仅浏览了开头部分（约 1 节），后续内容未逐段审阅。

## 开头可见的要点（据抓取文本，未经核实）

- 自回归适合低频、需因果推理的任务；扩散适合高频、实时性强的任务。
- 机器人场景中冻结 VLM 只微调动作头效果差，语义与控制被割裂（文中举 OpenVLA、Wall-OSS）。
- 动作空间：Delta 优于 Absolute；Joint 空间优于 Task Space（文中注明基于同本体、同数据量下的单篇论文结论，未考虑 EE 数据通常更多与 joint 数据对齐困难的问题）。
- 移动（locomotion）的 sim2real gap 相对易解决，操作（manipulation）的 gap 难解决。

## Robotics_Notebooks 映射

- 暂未升格为 wiki 页；若后续需要，可并入 [Sim2Real](../../wiki/overview/hub-sim2real.md) 相关页的讨论。

## 全文主题索引（2026-10-22，关键词级抽取）

对抓取文本（832 块，约 6.8 万字符）做标题与短句抽取，**未逐段通读**。文中出现的主题包括：

- **数据集：** Open X-Embodiment、BridgeData V2、DROID、RT-1、LIBERO、RoboMimic、Meta-World
- **运动学：** FK / IK 的概念与 VLA 中的位置（图像 + 文本 + 状态 → VLA → IK → 关节目标）
- **强化学习：** 奖励与 Q 值的递推示例、在线 HIL-RL（HIL-SERL）
- **生成模型：** 流匹配的起点与终点插值、DiT、LDM、I-JEPA、diffusion policy、CVAE
- **VLA 与动作表示：** RT-2、OpenVLA、CCT、π0（自回归与扩散两路）、DCT / BPE 动作 tokenization、动作块与时序融合
- **视频世界模型：** Cosmos 的扩散与离散两路（latent → decoder）
- **推理与训练方法：** RTC（实时分块）、SmolVLA、Wall-OSS、PLD（Probe / Learn / Distill）
- **记忆与价值：** π0.6 与 MEM 多尺度记忆、VLA + RECAP（价值与优势条件化，ACP）

以上为主题列表，具体结论与数字未逐条核对；文中部分内容是单篇论文的结论，文中已自行注明其适用条件有限。
