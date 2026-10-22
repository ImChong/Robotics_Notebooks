# Agensh 论文来源归档

## 元数据

- **题目：** Agensh: Scaling Organizational Intelligence to 1,024 Agents
- **作者：** Zhihao Zhan, Ting Song, Li Dong, Shaohan Huang, Jianxun Lian, Yan Xia, Furu Wei
- **单位：** Microsoft Research（据项目页）
- **预印本：** https://arxiv.org/abs/2609.26781（cs.CL，同时列 cs.MA；2026-09-22 提交）
- **PDF：** https://arxiv.org/pdf/2609.26781
- **HTML：** https://arxiv.org/html/2609.26781v1
- **项目页：** https://agens-harness.github.io/project/（含回放 demo）
- **短链：** https://aka.ms/Agensh → 301 跳转到项目页
- **会议 / 发布页：** https://www.microsoft.com/en-us/research/publication/agensh-scaling-organizational-intelligence-to-1024-agents/
- **研究院入口：** https://aka.ms/GeneralAI（本次未抓取，内容未核实）
- **代码：** https://github.com/microsoft/Agensh 直接抓取返回 **404**（仓库可能未公开或尚未创建，未能核实）；项目页未列代码
- **入库日期：** 2026-10-22

## 论文要解决的问题

大规模多 agent 协作通常依赖中心 orchestrator；论文探讨无中心编排时，agent 组织能否随规模扩大而提升效果。

## 方法要点（据 arXiv 摘要与项目页）

- 无中心 orchestrator 的多 agent harness：worker 异步协调，包括收集上下文、认领与分配子任务、执行、共享发现、验证结果并合并进度。
- 三个组件：共享工作区（追踪已提出、进行中、已完成的工作）、worker 间消息接口、存放可复用发现与工作意图的共享上下文。

## 报告的结果（据 arXiv 摘要）

- ProgramBench 最难的 5 个任务（GPT-5.6-sol high）：agent 从 1 扩展到 128，平均最终测试通过率 19.31% → 28.78%（约相对提升 49%）。
- pandoc 从零构建：agent 从 1 扩展到 1,024，最终通过率 33.89% → 55.06%；规模更大的组织更快达到相近通过率，协作模式随规模出现并趋于标准化。

## Robotics_Notebooks 映射

- 实体页：[Agensh](../../wiki/entities/paper-agensh.md)

## 一手来源

- arXiv 摘要页与项目页（经 WebFetch 摘要）
