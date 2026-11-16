# Compass 1 官方项目页

- **主页：** <https://www.archon.tech/research/compass1>
- **论文：** <https://arxiv.org/abs/2611.12087>
- **仓库：** <https://github.com/OpenDriveLab/Compass>
- **核验日期：** 2026-11-16
- **核对范围：** 官网 Introduction、Method、Data Pipeline、Evaluation 的正文、320 次主实验与消融/跨本体表、开放入口；互动演示视频未逐帧检查。
- **通读：** 已完成
- **开放程度：** 论文、项目页与展示仓公开；数据、部署代码和真机模型权重仍在 TODO，预计 2026-12 发布。

## 主要一手证据

- 方法：稀疏关键帧 + HRM；Task-Stage / Exploration / Spatiotemporal Retrieval 三类 VQA；单 GPU 异步推理中的 12 步动作生成、语言意图和停止。
- 真机：Go2 8 任务×4 场景×10 次 = 320；作者报告 254 成功（79.4%），AwareVLN 17.8%；轮式、人形、无人机独立各 10 次。
- 项目展示的结果属作者评估，不能代表公开 checkpoint 已可下载。

- **Wiki：** [Compass 1](../../wiki/entities/paper-compass-1-navigation.md)
- **源码状态：** [Compass 1 展示仓归档](../repos/compass1-release-status.md)
