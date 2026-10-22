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
