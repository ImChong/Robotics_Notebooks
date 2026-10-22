# UniWALM 论文来源归档

## 元数据

- **题目：** UniWALM: Unified World Action Language Model for Embodied Robot Control
- **作者：** Rahul Sajnani, Diego Romeres, Mingyuan Zhang, Kai Gao, Carl Qi, Baichuan Huang, Srinath Sridhar, Dylan Randle, Lillian J. Ratliff, Jose Barreiros, Manikantan Nambi, Jeroen Vanbaar
- **单位：** 抓取页面未列出
- **预印本：** https://arxiv.org/abs/2610.28626（cs.RO，2026-10-20 提交）
- **项目页 / 代码：** 本次抓取未发现项目页或代码链接（未核实是否存在）
- **入库日期：** 2026-10-22

## 论文要解决的问题

具身控制通常把世界预测、动作生成与语言规划分开处理；作者希望在同一骨干内统一三者，并让模型自身生成的语言作为记忆反馈，而不依赖外部规划器。

## 方法要点（据 arXiv 摘要）

- 同一个 mixture-of-transformers 骨干联合生成视频、动作与语言。
- 生成的语言以记忆形式回馈，无需外部 planner。
- **Flow Matching with Correction：** 训练目标，学习从扰动的动作轨迹中恢复，推理时无额外开销。

## 报告的结果（据 arXiv 摘要）

- MolmoSpaces pick-and-place：优于同骨干基线（如 FastWAM）。
- RMBench 记忆依赖子集：34.5%，高于 Mem-0 的 28.5%。
- 真机 Find-the-Item：加入文本记忆后，成功率 11.1% → 61.1%；摘要称当前 SOTA VLA 最高为 33.3%。

## Robotics_Notebooks 映射

- 实体页：[UniWALM](../../wiki/entities/paper-uniwalm.md)

## 一手来源

- arXiv 摘要页（经 WebFetch 摘要）
