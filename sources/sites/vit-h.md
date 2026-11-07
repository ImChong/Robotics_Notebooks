# ViT-H 官方项目页

> 来源归档（ingest）

- **类型：** site / humanoid control / research project
- **URL：** <https://awfuact.github.io/vit-h/>
- **论文：** <https://arxiv.org/abs/2610.32966> — [论文归档](../papers/vit_h_arxiv_2610_32966.md)
- **代码：** <https://github.com/mileret/ViT-H-release> — [仓库归档](../repos/mileret_vit_h_release.md)
- **入库日期：** 2026-10-27
- **复核日期：** 2026-11-08
- **核对范围：** 官方项目页完整公开内容（摘要、方法三阶段、真实部署、视频、代码链接及发布日期）；官方代码仓库 README 与关键运行入口。
- **通读：** 已完成
- **一句话说明：** ViT-H 项目页汇总从单目视频生成 G1 动作、跟踪反馈后训练和 G1 实机视频遥操作的演示与论文入口。

## 页面要点

- 项目采用生成与跟踪两阶段：视频条件扩散模型直接输出机器人空间参考动作，再由全身控制策略执行。
- 跟踪策略的执行结果用于构造候选动作偏好，给生成器做 DPO 后训练。
- 项目页展示离线 video-to-humanoid 控制和在线视频遥操作；平台为 Unitree G1，在线演示使用单台 RealSense D455 的 RGB 输入。
- 论文报告在线系统延迟低于 400 ms；项目页提供演示视频与论文入口。

## 开放状态核查（2026-11-08）

项目页所指的 [mileret/ViT-H-release](https://github.com/mileret/ViT-H-release) 已按计划开放离线视频到 G1 的推理与预训练实现，还包含 MuJoCo G1 跟踪器及 Hugging Face 模型/数据入口。

| 模块 | 状态 | 复核入口 |
| --- | --- | --- |
| 离线推理 | 已发布 | `tools/infer_video.py` |
| 离线预训练 | 已发布 | `scripts/train.sh` → `tools/train.py` |
| G1 MuJoCo 跟踪 | 已发布 | `g1_tracker/tools/track_g1.py` |
| 在线遥操作部署、在线预训练、DPO 后训练 | 计划 2026-12-01 发布，当前未见入口 | 仓库 README Release Plan |

论文示范使用 RealSense D455 的 RGB 输入；论文附录报告 RTX 5090 上平均总延迟 253.7 ms，但不代表已发布真实机器人在线部署代码。

## 关联资料

- 论文归档：[ViT-H arXiv:2610.32966](../papers/vit_h_arxiv_2610_32966.md)
- 官方仓库：[mileret/ViT-H-release](../repos/mileret_vit_h_release.md)
- Wiki：[ViT-H 论文实体页](../../wiki/entities/paper-vit-h-video-to-humanoid.md)
