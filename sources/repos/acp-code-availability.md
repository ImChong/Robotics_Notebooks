# ACP 代码与数据开放状态核查

- **项目：** Humanoid Mobility via Attention-based Contacts（ACP）
- **项目页：** <https://leggedrobotics.github.io/ACP/>
- **论文：** [arXiv:2610.31608](https://arxiv.org/abs/2610.31608)
- **核查日期：** 2026-10-23
- **结论：** ACP 项目页当前可读取，但未列该论文接触规划器、全身跟踪器或数据的下载链接。页面说明复用 AME-2 神经地图；其官方最小代码仓库已公开，不过 README 明确标注 Mid-360 神经地图模型不包含，神经地图代码列为后续发布计划。故可复用前作基础代码已公开，ACP 的接触规划/跟踪实现及神经地图模型仍无法从这些入口获取；这不等于作者确认不开放。

## 后续复核

后续若 ACP 项目页新增 Code / Resources 链接，复核源码与数据状态。AME-2 的仓库内容边界以[官方仓库 README](https://github.com/leggedrobotics/ame2_minimal)为准；其公开的训练代码不含神经地图模型。
