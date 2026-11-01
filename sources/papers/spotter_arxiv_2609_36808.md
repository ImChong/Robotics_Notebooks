# Spotter 论文来源归档

## 元数据

- **题目：** Spotter: Let the Embodied Model Lead, and the VLM Reflect for It
- **作者：** Long Li, Qichao Zhao, Yue Yang, Fan Xu, Zhe Wang, Alan Wee-Chung Liew, Chao Qu, Heng Tao Shen, Shirui Pan
- **单位：** Griffith University、清华大学 IIIS、MainCode、腾讯、复旦大学、同济大学（论文首页）
- **预印本：** https://arxiv.org/abs/2609.36808（cs.RO，also cs.AI；v1，2026-09-29 提交；19 页，7 图，6 表）
- **项目页：** https://zqc3117.github.io/Spotter/
- **代码：** https://github.com/zqc3117/Spotter（MIT；见 [代码归档](../repos/spotter.md)）
- **入库日期：** 2026-10-22

## 论文要解决的问题

具身模型自身的失败很少能自我恢复：重试很少修正错误，best-of-N 也难以可靠地选中成功的重试。作者归因于训练只用成功示范，且输入不足以暴露错误。

## 方法要点（据 arXiv 摘要）

- 与常见的 VLM 主导设计相反：具身模型持续执行（System 1），VLM（System 2）并行运行。
- 一个轻量本地 screener 并行检测错误；只有确认错误时 VLM 才介入，修复后将控制权交还。

## 报告的结果（据 arXiv 摘要）

- 以 GPT 作为 VLM：Cosmos Policy 在 RoboCasa 上 +5.6 分，π0.5 +7.5 分。
- RoboTwin 2.0 Hard 设置：π0.5 从 47.2% 提升到 57.0%。
- 真机：成功率 53% → 83%。
- 以 Qwen 作为 VLM：成功 episode 比具身模型单独执行仅多 13–16 秒；比 VLM 主导基线少约 70% 时间。

## Robotics_Notebooks 映射

- 实体页：[Spotter](../../wiki/entities/paper-spotter.md)
- 代码归档：[Spotter 代码仓](../repos/spotter.md)

## 一手来源

- arXiv 摘要页（经 WebFetch 摘要）；项目页因截断未能核对作者与数字

## 本次补充核查（2026-11-01，未完成）

- 已定位 [论文 PDF](https://arxiv.org/pdf/2609.36808) 和 [官方仓库](https://github.com/zqc3117/Spotter)；本次尚未完成全文及运行入口源码通读，须继续核对方法、实验与局限后绘制项目流程图和源码运行时序图。
- 官方 README 列出 RoboCasa 和 RoboTwin 2.0 评测入口及模型权重；项目页本次未能打开，需补核。
