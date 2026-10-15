# Cracking Mobile Manipulation: From Failure Diagnosis to Agentic Robot System（arXiv:2610.19600）

> 来源归档（ingest）

- **标题：** Cracking Mobile Manipulation: From Failure Diagnosis to Agentic Robot System
- **类型：** paper / mobile-manipulation / vla / long-horizon / failure-analysis / agentic / benchmark
- **arXiv：** <https://arxiv.org/abs/2610.19600>（PDF：<https://arxiv.org/pdf/2610.19600>；HTML：<https://arxiv.org/html/2610.19600v1>）
- **作者：** Zixuan Wang, Jia-Ning Li, Yinzhou Tang, Yanzhao Yu, Biao Wu, Meng Fang, Xin Li, Yiwu Zhong
- **机构（论文 HTML 作者单位）：** Peking University（School of Intelligence Science and Technology；State Key Lab of General AI）、Alibaba DAMO Academy / Alibaba Group、Institute of Automation, Chinese Academy of Sciences、Tsinghua University、University of Technology Sydney、University of Liverpool
- **代码（论文声明）：** <https://github.com/PKU-VaLuE-Lab/Cracking-Mobile-Manipulation>
- **开源核查（2026-10-15）：** **待发布**。仓库存在且公开（Apache-2.0），README 标注 "Coming soon"，列出六项计划发布内容（含训练/评测代码、检查点与 agentic 系统）；当前仓库仅含 PDF 预印本、LICENSE、README 与三张图，无运行脚本。

## 核心主张（摘要要点）

- 单任务 VLA 表现好，但长程移动操作差：预训练子任务 VLA 在原子操作上 80%，组合成移动操作后只有 19%。
- 失效诊断分三类：任务规划、导航与底盘定位、局部操作。
- 在固定子任务 VLA 外加四类失效导向干预：示范式规划引导、自适应视觉记忆、可达性感知底盘调整、局部末端调整。
- 结果：RoboCasa365 上 19% → 47%（开发集），held-out 种子上最终系统 49%，基线 18%；BEHAVIOR-1K 14% → 37%；真机 5/20 → 13/20。
- 结论：针对当前瓶颈的干预，可能比统一放大策略更有效。

## 平台与数据

- 真机：双臂移动操作平台（底盘与机械臂具体型号未核实），2 个任务、20 次试验。
- 仿真：RoboCasa365（32 个组合任务中选 20 个，每任务 5 次开发 rollout + 5 个 held-out 种子）；BEHAVIOR-1K（5 个任务，迁移测试）。

## 链接

- arXiv 摘要：<https://arxiv.org/abs/2610.19600>
- 代码仓库（待发布）：<https://github.com/PKU-VaLuE-Lab/Cracking-Mobile-Manipulation>
- wiki 节点：[paper-cracking-mobile-manipulation](../../wiki/entities/paper-cracking-mobile-manipulation.md)
