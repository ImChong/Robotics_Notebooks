# DevGRU 官方项目页

> 来源归档（ingest · 项目页核查）

- **标题：** DevGRU: Depth-guided Visual Navigation using a Collision-aware Recurrent Model
- **类型：** site / project-page
- **官方入口：** <https://graphics.ewha.ac.kr/projects/details/devgru/>
- **论文：** [arXiv:2608.18470](https://arxiv.org/abs/2608.18470)；RA-L 2026；DOI: 10.1109/LRA.2026.3730213
- **代码：** [EwhaGlab/devgru](https://github.com/EwhaGlab/devgru)（项目页 Code 按钮）— [代码归档](../repos/devgru.md)
- **权重与数据：** GitHub README 提供预训练模型权重和 mixed_20K 数据集的 Dropbox 下载入口。
- **机构：** 梨花女子大学（Ewha Womans University）
- **作者：** Kyung Min Han、Eunsom Kim、Young J. Kim（通讯作者）
- **入库日期：** 2026-10-17
- **一句话说明：** DevGRU 通过深度观测驱动的循环策略、碰撞预测与拓扑子目标修正，实现室内点目标长程导航。

---

## 项目页核查

- 页面提供 Paper 与 Code 入口；Code 链接指向公开 GitHub 仓库。
- AP 预测碰撞安全的未来路点和修正子目标位姿；CP 估计即将发生碰撞的概率，并在风险较高时将位姿修正传给后续子目标。
- 页面报告九张未见拓扑图、六个校园地点实验：平均 NC 0.89、SPL 0.80；19.44M 参数，推理 50.01 ms。
- 代码、预训练权重和数据集均有公开入口。代码许可证仅允许教育、研究和非营利用途；商业使用需另行取得许可。
- 仓库 README 指定 Ubuntu 20.04、ROS Noetic、Conda；复现还依赖 NavDataCollector 与机器人 ROS 配置。
