## [2026-11-10] ingest | EgoWild2Dex | 既有节点论文全文与附录精读升级

- **意图：** 避免与历史 PR #2296 的 EgoWild2Dex 同项目节点重复，沿用唯一论文实体，补充原始文献通读证据。
- **原文：** arXiv:2609.23755v1（§I–VI，Algorithm 1，附录 A–D，图表）与 https://mmlab.hk/egowild2dex/。
- **研究事实：** EgoWild 538.9 h、179,049 episodes；GeoFormer 五级 8 参数单应变换，学 robot→ego，使用逆变换对齐 ego；三阶段 FM VLA（task-specific ego、robot-view glove、expert robot + 10 条 DAgger）。
- **工程边界：** 96.7% 是三任务各 10 次共 29/30 次真机成功；图像对齐 21.9× 不是控制策略加速；每任务 robot <1 h 不包含人类数据与 A100 训练算力。
- **开放核查：** 2026-11-10 项目页无可下载的 EgoWild/GeoFormer 官方实现与模型资源，论文仅承诺待发布，源码时序图不适用。
- **回写：** [EgoWild2Dex](../wiki/entities/paper-egowild2dex.md)；[论文源归档](../sources/papers/egowild2dex_arxiv_2609_23755.md)；[项目页源归档](../sources/sites/mmlab-egowild2dex.md)。
