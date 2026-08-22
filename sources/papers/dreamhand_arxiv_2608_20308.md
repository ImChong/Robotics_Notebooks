# DreamHand: Repurposing Video Diffusion Models for Occlusion-Robust Egocentric 3D Hand Motion Recovery

> 来源归档（ingest）

- **标题：** DreamHand: Repurposing Video Diffusion Models for Occlusion-Robust Egocentric 3D Hand Motion Recovery
- **短名：** DreamHand
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.20308>
- **PDF：** <https://arxiv.org/pdf/2608.20308>
- **项目页：** <https://ggxxii.github.io/dreamhand/>
- **代码：** <https://github.com/ggxxii/dreamhand>
- **入库日期：** 2026-08-22
- **索引来源：** [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md)（<https://mp.weixin.qq.com/s/EmC4gNgcQdPX34vxy-qSVQ>）
- **一句话说明：** 见下方摘录与 wiki 映射。

## 开源状态（步骤 2.5）

- **待发布**：[`ggxxii/dreamhand`](https://github.com/ggxxii/dreamhand) 仓已建；README 写明推理/权重/训练脚本 **尚未发布**（watch 更新）。

## 核心摘录（面向 wiki 编译）

### 摘录 1：VDM 作几何编码器

- 单次 clean latent 前向暴露遮挡/出画场景内容；双向时空解码器恢复连续双手轨迹；Ray-Based Camera Solver 支持无测试时内参。

**对 wiki 的映射：** egocentric-vision、human-video-to-robot

### 摘录 2：基准

- 五 egocentric benchmark SOTA；ARCTIC/HOT3D MPJPE-p ↓30%/40%，含出画手收益 46%–61%。

**对 wiki 的映射：** 手部重建、操作数据扩展

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-dreamhand.md`](../../wiki/entities/paper-dreamhand.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与技术地图回链
