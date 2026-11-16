# MATE（原 LAG-Fusion）官方项目页

> 来源归档（site）

- **项目名称：** MATE: Asynchronous Multi-Rate Diffusion Policy Composition
- **项目主页：** <https://lag-fusion.github.io/>
- **主论文：** <https://arxiv.org/html/2607.17257v2>（2026-11-06 v2）
- **旧名称：** LAG-Fusion: Asynchronous Multimodal Diffusion Policy Composition via Latency-Aware Guidance Fusion
- **核查日期：** 2026-11-10
- **代码状态：** 项目页面只显示 `code coming soon`，**尚无可核验的官方源码 URL**，因此未建立 `sources/repos/` 归档。
- **数据/权重状态：** 项目页展示视频与评测表，但未给公开下载入口；不应视作训练数据已开放。
- **一句话说明：** 独立扩散分支异步组合：视觉全局引导、力觉快速纠偏，也支持全局 3D / 腕部 2D 异步视觉组合。

## 项目页核查

- Vision–force：Flip 和 Polish Curve，视觉 5 Hz、力觉 25 Hz。
- Vision–vision：遮挡抓取放置任务，全局 3D 2 Hz、腕部 2D 5 Hz。
- 展示参考帧重映射：Overlap → Rebase → Denoise → Fuse → Map back。
- 原始论文与项目页旧名在检索索引中仍可能出现；同一 arXiv 2607.17257 为 v1 → v2 **改名**，不要重复建实体节点。

## 对 wiki 的映射

- [MATE 唯一实体节点](../../wiki/entities/paper-mate-asynchronous-multirate-diffusion.md)
- [论文通读归档](../papers/mate_asynchronous_multirate_diffusion_arxiv_2607_17257.md)
