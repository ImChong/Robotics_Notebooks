# DeliverBench｜官方项目页

> 来源归档（官方站点）

- **标题：** DeliverBench: Benchmarking AI Agents on Delivering Realistic Multi-Stage Professional Projects
- **类型：** site
- **链接：** <https://deliverbench.github.io/>
- **论文：** <https://arxiv.org/abs/2611.03412>
- **代码：** <https://github.com/theseus-labs-rsi/deliverbench>（已公开，范围见代码归档）
- **数据：** <https://huggingface.co/datasets/deliverbench/DeliverBench>
- **入库日期：** 2026-11-09
- **核对范围：** 官方网站的项目说明、论文/代码与数据入口；配合代码主仓 README 验证开放范围。
- **通读：** 未完成

## 关键信息

- 论文研究对象是持续处理专业项目的 agent，不是单次答题或单个 coding issue。研究集合包含 **11 个项目、7 个领域、150 个阶段任务**；公开数据集当前仅完整开放 **3 个案例**，与论文总研究集不可混同。
- 公开案例：Mars Refinancing（13 阶段）、Pharmacovigilance ICSR（12 阶段）和 StarPulse Legal Delivery（8 阶段）。
- Agent 在 **同一持续工作空间** 中接收阶段材料、用户反馈并修改早先产物；开场背景不计分，阶段评测和独立 FINAL 评测区分。
- 每个可计分单元的证据冻结；判分器对 criterion 作完整性检查，真正违反 hurdle 才可有效记零，遗漏判决/不可读证据/运行错误不伪装成零分。
- 论文比较多种 agent 模型与 harness，但模型分数属于具体裁判、预算和项目集合的组合，不能与只开放三例的复现实验混称。

## 开放与复现边界

三例公开数据不覆盖论文中的全部研究项目；运行需要 Python 3.12、Harbor 0.20.0、Docker 与候选/Judge/Responder 配置；授权材料和未来输入必须对被评 agent 隐藏。

## 对 wiki 的映射

- [DeliverBench 主实体](../../wiki/entities/paper-deliverbench.md)
- [论文归档](../papers/deliverbench_arxiv_2611_03412.md)
- [代码仓归档](../repos/deliverbench.md)
