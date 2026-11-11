# ASPIRE（arXiv:2611.13114）

> 来源归档（paper）

- **标题：** ASPIRE: An Agentic System-Identification Platform for Iterative Reasoning and Exploration
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.13114>
- **PDF：** <https://arxiv.org/pdf/2611.13114>
- **代码：** https://github.com/erik-derner/aspire-sysid
- **作者 / 机构：** Erik Derner、Jiří Kubalík（捷克理工大学 CIIRC）、Robert Babuška（CIIRC 与代尔夫特理工大学）；arXiv v1（cs.RO），2026-11-09
- **入库日期：** 2026-11-11
- **一句话说明：** 让 LLM 智能体自选辨识方法、从实测数据构造闭式非线性模型，由外围 harness 封存测试集、独立核验形式化物理约束并留存全过程记录。
- **通读：** 已完成
- **核对范围：** 读了 arXiv v1 PDF 正文第 1-8 节（含 Table 1-7）、附录 A（系统提示）、附录 B 的文字与模型公式、附录 C（燃料发酵罐物理要求）、数据 / 代码可用性说明。图 2-5 只读了图注与正文描述，未逐点检视曲线；附录 B 参数表的全部数值未逐项核对。仓库只静态读了 README（未运行代码、未读 experiments 目录里的运行记录）。

## 开源状态

- 论文脚注 2 与「数据 / 代码可用性」称 harness、任务说明与数据、全部运行记录、PySINDy/PySR 对照脚本、记录评估证据均在 https://github.com/erik-derner/aspire-sysid（CC BY 4.0）。README 静态读过，与论文描述一致；README 自述「预印本链接稍后补充」。

## 核心摘录

1. **问题与假设（第 3 节）：** 静态、连续 / 离散状态空间与 NARX 的闭式模型；假设 A1 评分输出为标量，A2 状态可测，A3 测量方程给定。
2. **框架（第 4 节）：** harness + 一或多个 LLM 智能体；测试数据封存，智能体工具无法触达；默认预算 60 轮 / 3600 s / 10 美元；九类形式化物理要求（Table 2）由 harness 符号或采样核验，有证明违反则剔除出排名与推荐；智能体推荐在测试结果出现前记录。
3. **模型记录（4.3）：** 参数表（fitted / fixed / tied）、估计记录、讨论；harness 只查完整性与一致性，不查内容（4.3）。
4. **实验（第 5 节）：** 三个实测系统（TurtleBot 2、干摩擦转动盘、40 L 发酵罐顶空压力），各有「有描述 / 盲」两种任务，每任务 5 次，共 30 次；智能体为 gpt-6.1-sol（medium 推理），2026-10-07 至 09；对照 PySINDy 与 PySR。
5. **结果（第 6 节）：** 30 次全部交付有效模型；126 个候选全部合格；单次成本 0.96-3.38 美元，30 次共 61.10 美元（6.4）；记录评估 150 项中 132 yes / 18 partly / 0 no，9 条记录的结构来源陈述与执行日志有一处出入（6.3、7.2）。
6. **局限（7.3、第 8 节）：** 仅单一语言模型且无日期快照；只支持单输出、状态可测；作者在开发中用过测试数据；交互评估无独立用户。

## 对 wiki 的映射

- [paper-aspire](../../wiki/entities/paper-aspire.md)
