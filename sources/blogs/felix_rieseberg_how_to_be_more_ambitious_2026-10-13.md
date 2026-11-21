# How to be more ambitious: A tutorial for humans with models

> 来源归档；保留已有同一原文身份，不重复创建实体。

- **类型：** blog（作者个人博客）
- **作者：** Felix Rieseberg（Anthropic）
- **链接：** https://felixrieseberg.com/how-to-be-more-ambitious-a-tutorial-for-humans-with-models/
- **发布日期：** 2026-10-13
- **入库日期：** 2026-10-28（本次复核：2026-11-21）
- **核对范围：** 通读 Speed、Altitude、Depth、Range、We have been here before 等全部正文段落、操作建议和示例；引用观点按作者原文，不将案例扩展推论冒充实证。
- **通读：** 已完成
- **一句话说明：** 把使用模型的野心分为速度、高度、深度、广度；核心是从产物转向最终效果、用真实专业方法、跨学科工具与外部验证。
- **沉淀到 wiki：** [Agentic Coding 时代的软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)；[Agent-first 软件工厂与 Harness Engineering](../../wiki/concepts/agentic-software-factory-harness.md)

## 四类雄心

- **Speed（速度）：** 同样的工作更快完成；这是普遍用法，作者主要讨论另外三类。
- **Altitude（高度）：** 从“修一次 crash”提升为“持续保持应用无 crash”；从“生成季度报告”提升为“提前发现需要调资源的业务问题”。要让模型真实实现效果，需提供 crash 数据、版本映射、可运行的系统和验证权。
- **Depth（深度）：** 让 agent 使用该领域最好的数据源、数据库、统计工具与检验方法，而不仅模仿专家词汇；挑战未经证明的业务假设、挖掘未知信号，用**独立的外部评判器**而非模型自评来验证。
- **Range（广度）：** 把名词形式的交付物重写成动词化目标，找需要的其他专业方法；例如不只要求“漂亮的 slide”，而是引入分析、编辑、红队和设计等可执行技术。不是让模型“扮演”不同岗位便算跨界。

## 核心工具与局限

作者建议模型拥有可安装软件的计算机和真实数据/工具访问（可经 MCP connector 或其他方式）；否则要求更高抽象目标只会停在文本规划。文末心理学与历史叙述用于说明“高目标”的直觉，不应由此推断模型自主性或单个案例的因果效应。

对机器人开发的**应用推论**：把“写一段 PPO 脚本”提升为“持续定位失稳原因”；先接入可复位仿真、策略指标和可检验的 rollout，再让 agent 自动分析与修正。真机上必须保留独立安全约束，这不是 Felix 文章的实证结果。

## 官方来源

- [Felix Rieseberg 原文（2026-10-13）](https://felixrieseberg.com/how-to-be-more-ambitious-a-tutorial-for-humans-with-models/)
