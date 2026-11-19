# MimicDroid 项目页

- **论文：** [arXiv:2509.09769](https://arxiv.org/abs/2509.09769)（ICRA 2026 Oral, Top-5%）
- **项目页：** <https://ut-austin-rpl.github.io/MimicDroid/>
- **基准代码：** <https://github.com/UT-Austin-RPL/mimicdroid-robocasa/tree/latest>
- **数据集：** <https://huggingface.co/datasets/Rutav/MimicDroidDataset>
- **机构：** UT Austin; Amazon Consumer Robotics; NVIDIA
- **入库日期：** 2026-11-19
- **核对范围：** 通读项目页全文（ICRA 2026 Oral 录用信息、Meta-ICL 示意图、Top-K 检索可视化、L1–L3 仿真对比、GR1 与 G1 真机视频、失败案例卡片及 BibTeX）。
- **通读：** 已完成

## 开源与公开资源核查（2026-11-19）

| 组件 | 状态 | 详情 |
|------|------|------|
| 项目页 | 有 | 展示 ICRA 2026 Oral (Top-5%) 录用信息、GR1 与后续 `[New!] Rollouts on G1 Robot` 真机演示与失败归因视频 |
| 仿真基准环境 | **已开源** | [UT-Austin-RPL/mimicdroid-robocasa](https://github.com/UT-Austin-RPL/mimicdroid-robocasa/tree/latest)（基于 RoboCasa，含 L1–L3 任务及回放入口） |
| Play 与任务数据集 | **已发布** | Hugging Face `Rutav/MimicDroidDataset`（约 8 GB 含 `training/` 与 `TaskDemos/`） |
| ICL 策略训练/检索源码 | **未公开发布** | 经 2026-11-19 核查代码仓，`mimicdroid-robocasa` 仅包含基准环境与数据集下载/回放脚本，不含 WiLoR+DINOv2 检索与 ICRT 训练代码 |

## 交叉

- 论文归档：[`sources/papers/mimicdroid_arxiv_2509_09769.md`](../papers/mimicdroid_arxiv_2509_09769.md)
- 代码仓归档：[`sources/repos/ut-austin-rpl-mimicdroid-robocasa.md`](../repos/ut-austin-rpl-mimicdroid-robocasa.md)
- wiki：[`wiki/entities/paper-notebook-mimicdroid-in-context-learning-for-humanoid-robo.md`](../../wiki/entities/paper-notebook-mimicdroid-in-context-learning-for-humanoid-robo.md)
