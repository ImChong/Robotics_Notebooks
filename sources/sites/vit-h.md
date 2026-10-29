# ViT-H 官方项目页

> 来源归档（ingest）

- **类型：** site / humanoid control / research project
- **URL：** <https://awfuact.github.io/vit-h/>
- **论文：** <https://arxiv.org/abs/2610.32966> — [论文归档](../papers/vit_h_arxiv_2610_32966.md)
- **代码：** <https://github.com/mileret/ViT-H-release> — [仓库归档](../repos/mileret_vit_h_release.md)
- **入库日期：** 2026-10-27
- **一句话说明：** ViT-H 项目页汇总从单目视频生成 G1 动作、跟踪反馈后训练和 G1 实机视频遥操作的演示与论文入口。

## 页面要点

- 项目采用生成与跟踪两阶段：视频条件扩散模型直接输出机器人空间参考动作，再由全身控制策略执行。
- 跟踪策略的执行结果用于构造候选动作偏好，给生成器做 DPO 后训练。
- 项目页展示离线 video-to-humanoid 控制和在线视频遥操作；平台为 Unitree G1，在线演示使用单台 RealSense D455 的 RGB 输入。
- 论文报告在线系统延迟低于 400 ms；项目页提供演示视频与论文入口。

## 开放状态核查

项目页链接到官方 GitHub 实现仓库。核对该仓库 README：截至 2026-10-27，仓库公开但尚未提供可运行代码；README 将离线控制和实时遥操作实现列为未来发布计划。项目演示和开源计划不能视为已发布可复现代码。

## 关联资料

- 论文归档：[ViT-H arXiv:2610.32966](../papers/vit_h_arxiv_2610_32966.md)
- 官方仓库：[mileret/ViT-H-release](../repos/mileret_vit_h_release.md)
- Wiki：[ViT-H 论文实体页](../../wiki/entities/paper-vit-h-video-to-humanoid.md)
