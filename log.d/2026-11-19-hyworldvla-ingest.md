# 2026-11-19 Ingest：HyWorldVLA 自动驾驶混合世界建模

- **操作：** ingest 论文 arXiv:2607.20988（Quanfu Yu、Xian Wu、Hao Xu、Liulong Ma，比亚迪汽车新技术研究院）。
- **一手资料：** https://arxiv.org/html/2607.20988v1
- **原文归档：** [sources/papers/hyworldvla_arxiv_2607_20988.md](../sources/papers/hyworldvla_arxiv_2607_20988.md)
- **唯一论文实体：** [wiki/entities/paper-hyworldvla.md](../wiki/entities/paper-hyworldvla.md)
- **核对：** 论文正文、实验表、附录失败案例与损失权重消融；把 NAVSIM v1 PDMS 与 v2 EPDMS 分开，并区分正常天气与 655 例雨雾噪声子集。
- **架构要点：** 预训练的像素与 latent 混合监督 → 联合微调阶段让未来 latent 条件化规划头；流程图只画原文方法，不冒充运行时源码图。
- **关联：** 在 S²-VLA 详情增加方法对照与入链；挂载至 VLA Stage 5「世界模型融合」。
- **开放性：** 论文未提供 HyWorldVLA 官方实现和权重链接，因此不编造运行时调用、训练日志或复现结论。
- **验证：** 待 GitHub Actions Wiki Lint / Tests / Search & Export Quality 检查；未在本运行环境完整 clone 仓库跑 make ci-preflight 或真实驾驶仿真。
