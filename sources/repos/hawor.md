# HaWoR（egocentric 世界系手部运动重建）

> 来源归档（repo）

- **标题：** HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos
- **类型：** repo / paper implementation
- **项目页：** <https://hawor-project.github.io/>
- **代码：** <https://github.com/ThunderVVV/HaWoR>
- **论文：** <https://arxiv.org/abs/2501.02973>（CVPR 2025 Highlight）
- **机构：** 上海交通大学 / Imperial College London
- **入库日期：** 2026-08-07
- **一句话说明：** 从 egocentric RGB 视频解耦 **相机系时序手重建（MANO）** 与 **世界系相机轨迹（自适应 egocentric SLAM + 度量对齐）**，并用 motion infiller 补全视野外帧，输出世界坐标下的双手轨迹。

## 开源状态（步骤 2.5，截至 2026-08-07）

- **已开源：** 官方 GitHub [ThunderVVV/HaWoR](https://github.com/ThunderVVV/HaWoR)；项目页链到代码与 arXiv。
- **许可：** GitHub 元数据未标注标准 SPDX（`NOASSERTION`）；复现以仓库 LICENSE / README 为准。
- **角色边界：** Macrodata 手部动作博客以 HaWoR 为 **相机系手重建** 强基线，并用 **VGGT-Omega 窗口拼接** 替换其原 DROID-SLAM+Metric3D 世界重建以提速；见 [macrodata_egocentric_video_3d_hand_actions.md](../blogs/macrodata_egocentric_video_3d_hand_actions.md)。

## 核心模块（项目页 / 论文摘要）

1. **检测 → 相机系时序重建：** 离架检测器裁剪后，大尺度 Transformer + 运动先验预测 MANO 与腕位姿。
2. **自适应 egocentric SLAM：** 手部像素掩蔽后估计相机轨迹，并用 foundation metric depth 做尺度对齐。
3. **Motion infiller：** 补全出视野 / 低置信间隙（Macrodata HOT3D 评测显示其 learned infiller 未必优于线性插值基准规则）。

## 对 wiki 的映射

- [macrodata-egocentric-hand-action](../../wiki/methods/macrodata-egocentric-hand-action.md) — 开源管线选型与 Action MPJPE 工程消融
- [WiLoR](../../wiki/methods/wilor.md) — 常用检测前端
- [ViDiHand](../../wiki/entities/paper-vidihand.md) — 对照：video diffusion 先验、无 detector 的 egocentric 双手 4D

## 当前提炼状态

- [x] 项目页 / GitHub / arXiv 入口与开源结论
- [x] 与 Macrodata 博客配方交叉索引
- [ ] 未单独升格 HaWoR 论文实体页（本次以 Macrodata 工程博客为主）
