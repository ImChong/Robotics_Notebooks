# WorldFlow 官方项目主页

> 来源归档（site）

- **类型：** site
- **标题：** WorldFlow: Joint Video and 3D Motion Prediction via Large-Scale Pretraining
- **项目主页：** <https://worldflow-rbs.github.io/>
- **论文：** <https://arxiv.org/html/2610.39278v2>
- **项目页 Code 按钮：** <https://github.com/roboscience-ai/worldflow>
- **原入库：** 2026-10-30
- **复核：** 2026-11-20
- **沉淀到 wiki：** [WorldFlow](../../wiki/entities/paper-worldflow.md)
- **源码状态详细核查：** [官方资源仓库](../repos/worldflow.md)

## 页面核对

主页公开模型动机、联合视频/3D flow MoT 架构、WorldFlowBench 定量结果与仿真/真机展示。摘要称约 144 万条物体级训练样本、1,678 条测试样本；SpatialCog ADE 0.060 m。

页面虽然提供 Code / Data / Model 资源区入口，但**不能仅以存在按钮判定权重、数据或训练/推理代码可下载**。本次按 Code 链接实际到达的 GitHub README Release Plan 核查，仓库目前仅含论文、说明与视觉素材；模型实现、checkpoint 和 benchmark 工具待发布。详情见 [论文原文核对](../papers/worldflow_arxiv_2610_39278.md)。

## 工程边界

该研究的模型输出为 3D 运动预测，真机演示还使用目标分割、抓取、轨迹优化和机器人控制。项目展示成功率不可误写成开源模型可复现的部署结果。
