# ERR: Replan for Error Recovery 论文来源归档

## 元数据

- **题目：** Replan for Error Recovery: Training Embodied VLMs through Verifiable Interaction with Action-Grouped Turn-GSPO（方法简称 ERR / AGT-GSPO）
- **作者：** Zhuoran Wang, Keqing Fan, Yang Li, Jinbo Su, Ke Wang, Sijia Luo, Yuanzheng Tan, Junken Peng, Jing Zhang
- **单位：** 抓取页面未列出
- **预印本：** https://arxiv.org/abs/2610.27003（cs.RO，v1，2026-10-19）
- **项目页 / 代码：** 抓取页面未列出
- **入库日期：** 2026-10-22

## 论文要解决的问题

具身视觉语言模型（VLM）执行复杂操作时，单步执行失败会使环境进入需要重新规划的状态。

## 方法要点（据 arXiv 摘要）

- **ERR-Env：** 给出任务进度与错误反馈的交互环境。
- **ERR-Bench：** 衡量错误恢复能力与执行可靠性的基准。
- **AGT-GSPO：** 在线 RL，使用 turn 级奖励与 action-grouped 优势估计。
- **ERR-VLM：** 用 AGT-GSPO 训练的 2B 紧凑模型。

## 报告的结果（据 arXiv 摘要）

- 不同 VLM 之间，任务进度与执行可靠性并不一致。
- AGT-GSPO 使 ERR-VLM 的 Strict Success Rate 提升 10.25 点，Recovery Rate 提升 22.95 点。

## Robotics_Notebooks 映射

- 实体页：[ERR（Replan for Error Recovery）](../../wiki/entities/paper-err-replan-error-recovery.md)

## 一手来源

- arXiv 摘要页（经 WebFetch 摘要）
