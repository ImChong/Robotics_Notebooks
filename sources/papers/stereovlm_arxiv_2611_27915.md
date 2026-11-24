# StereoVLM：度量三维空间推理的双目视觉语言模型（Binocular Vision-Language Models for Metric 3D Spatial Reasoning） — papers 来源归档

- **类型：** paper
- **标题：** StereoVLM: Binocular Vision-Language Models for Metric 3D Spatial Reasoning
- **机构：** 香港科技大学、小鹏机器人
- **原文：** https://arxiv.org/html/2611.27915
- **arXiv：** 2611.27915
- **发表日期：** 2026-11-20
- **项目主页：** https://xpeng-robotics.github.io/stereovlm/
- **开放状态：** 截至 2026-11-24 项目页 Code / Models 未提供可点击公开链接；未确认公开实现或权重。
- **入库日期：** 2026-11-24
- **核对范围：** 已打开 HTML 全文及章节目录、核对项目页和主要指标；尚未逐节通读正文和附录。
- **通读：** 未完成
- **对应知识页：** [stereovlm](../../wiki/entities/paper-stereovlm.md)

## 资料要点

- 用左右图像与标定参数建立点级跨视图对应、视差和度量深度监督，再引入语言定位的物体级任务。对整流双目，深度与视差、焦距、基线存在几何约束；模型学习利用双目证据而不是仅凭单图语义猜尺度。Stereo3D-Bench 覆盖点级与物体级共八项任务，并用右视图干预检验真实双目贡献。
- 项目页报告 StereoVLM-8B 深度 δ₁ 95.17%、物体级均分 66.98。匹配的 geometry-only 训练中，未见场景的视差监督深度 δ₁ 74.92%，直接深度监督 45.89%。五个空间基准平均 +3.54 分；相同 GR00T 架构及双目输入设置下 RoboCasa 成功率 38.1%→42.4%。
- 依赖标定和双目观测，错误外参、遮挡、纹理不足等需单独检查；RoboCasa 为仿真，不是实机泛化证明。代码与权重未确认发布，暂不能完整复现。

## 公开资源

- [论文 HTML](https://arxiv.org/html/2611.27915)
- [arXiv](https://arxiv.org/abs/2611.27915)
- [官方项目页](https://xpeng-robotics.github.io/stereovlm/)

## 对 wiki 的映射

- [stereovlm 独立实体](../../wiki/entities/paper-stereovlm.md)
- [关联归档](../sites/stereovlm.md)
