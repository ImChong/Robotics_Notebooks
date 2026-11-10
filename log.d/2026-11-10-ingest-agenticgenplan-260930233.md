## [2026-11-10] ingest | sources/papers/agenticgenplan_coding_agents_generalized_tamp_arxiv_2609_30233.md — 通读 AgenticGenPlan 原文，核对官方项目与 RoboCode 运行入口，建立英中双语 TAMP 泛化节点、Mermaid 主流程图和源码时序图；澄清“同一环境内不同实例”不等于跨模拟器迁移。

- 一手材料：arXiv:2609.30233v1 官方 HTML、agenticgentamp.github.io、tomsilver/robocode@d3d42dd、作者 Hugging Face 运行数据。
- 开源状态：研究代码和运行数据已公开；作者论文 held-out seed 仍需按需申请。
- 规模核对：28 环境、980 个程序、98,000 次评估；16 个 planner 环境子集为 47% 对 Astra 95%；动态 3D 仍有明显弱项。
- 防重复：论文、项目站点与官方源码归同一 wiki 项目实体；仅新增源文件与 log.d，不提交 catalog.md / log.md。
