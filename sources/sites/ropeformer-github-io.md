# RopeFormer 项目页（ropeformer.github.io）

> 来源归档（site）

- **标题：** RopeFormer: Cross-Trial Adaptation from Interaction History for Dynamic Rope Manipulation
- **类型：** project-page
- **URL：** <https://ropeformer.github.io/>
- **论文：** [arXiv:2609.23432](../papers/ropeformer_arxiv_2609_23432.md)
- **机构：** UC Berkeley MSC Lab；Xi'an Jiaotong University；SUSTech；Peking University
- **入库日期：** 2026-09-30
- **代码：** **待发布** — 导航 **Code SOON**（`aria-disabled="true"`，2026-09-30）
- **Paper：** 链至 [arXiv:2609.23432](https://arxiv.org/abs/2609.23432)
- **一句话说明：** 跨 trial Transformer-XL 记忆 + 三任务仿真/真机视频与 Fig.3–9 级结果叙述。

## 核查结论（步骤 2.5）

- **已公开：** Paper/arXiv、Full 方法动画、三任务 one-shot / repeated-trial 视频、仿真 384 绳曲线、H1-2 真机块
- **待发布：** Code（SOON tag）；未见 GitHub / Hugging Face 直链
- **摘要 vs 页：** arXiv 写 code/data at 项目页 — 以页上 **SOON** 为准至仓库上线

## 页面要点摘录

- **Episode vs trial：** 同绳多 trial 保留 policy context；新 episode 清 context
- **架构：** 6-layer TXL，segment **L=128**，KV cache 流式 30 Hz
- **训练：** Newton；PPO 非对称 critic（特权绳/执行器参数）；trial 边界截断 GAE
- **观测：** TXL-1 / TXL-6（1 或 6 绳点）vs MLP-8 帧（无 cross-trial）
