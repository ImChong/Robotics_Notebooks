# DeliverBench: Benchmarking AI Agents on Continuous Professional Project Delivery｜面向连续专业项目交付的智能体评测

> 原论文归档（截至 2026-11-09：核查过论文摘要、官方项目说明和代码 README；全文逐节核对尚未完成）

- **类型：** paper
- **原文：** <https://arxiv.org/abs/2611.03412>
- **项目：** <https://deliverbench.github.io/>
- **代码：** <https://github.com/theseus-labs-rsi/deliverbench>
- **入库日期：** 2026-11-09
- **核对范围：** 论文摘要与方法/评测选段、官方项目页、主仓 README 及相关运行入口；附录和全文尚未逐页完成核对。
- **通读：** 未完成

## 研究问题与核心事实

- **论文：** <https://arxiv.org/abs/2611.03412>（2026-11 公开版本；版本可能更新）
- **研究范围：** 11 项项目、7 类专业领域、150 个阶段任务；公开仓库 README 只提供 3 个完整项目，须分开核算。
- **问题：** 单次 agent 基准不能衡量跨阶段交付的上下文保持、追溯、修订与最终验收。
- **评测：** 阶段 process 与最终 FINAL 独立判分；rubric 带 criterion、weight、hurdle、dependency，冻结当时证据。
- **局限：** 多数研究案例未作为公开复现数据交付；Judge 标定与项目选择可影响可比性。

## 方法与评测摘录

- 论文研究对象是持续处理专业项目的 agent，不是单次答题或单个 coding issue。研究集合包含 **11 个项目、7 个领域、150 个阶段任务**；公开数据集当前仅完整开放 **3 个案例**，与论文总研究集不可混同。
- 公开案例：Mars Refinancing（13 阶段）、Pharmacovigilance ICSR（12 阶段）和 StarPulse Legal Delivery（8 阶段）。
- Agent 在 **同一持续工作空间** 中接收阶段材料、用户反馈并修改早先产物；开场背景不计分，阶段评测和独立 FINAL 评测区分。
- 每个可计分单元的证据冻结；判分器对 criterion 作完整性检查，真正违反 hurdle 才可有效记零，遗漏判决/不可读证据/运行错误不伪装成零分。
- 论文比较多种 agent 模型与 harness，但模型分数属于具体裁判、预算和项目集合的组合，不能与只开放三例的复现实验混称。

## 复现与保留意见

三例公开数据不覆盖论文中的全部研究项目；运行需要 Python 3.12、Harbor 0.20.0、Docker 与候选/Judge/Responder 配置；授权材料和未来输入必须对被评 agent 隐藏。

本归档将已查过的一手材料与尚待全文复核的结论分开；未在本次会话实际运行论文实验，不应把代码入口的存在解释为分数已复现。

## 对 wiki 的映射

- [DeliverBench 主实体](../../wiki/entities/paper-deliverbench.md)
- [官方站点归档](../sites/deliverbench.md)
- [代码仓归档](../repos/deliverbench.md)
