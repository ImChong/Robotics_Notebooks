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
- **核对范围：** 论文正文与附录 A–D；项目主页及代码公开状态核对。
- **通读：** 已完成
- **对应知识页：** [stereovlm](../../wiki/entities/paper-stereovlm.md)

## 资料要点

- 用左右图像与标定参数建立点级跨视图对应、视差和度量深度监督，再引入语言定位的物体级任务。对整流双目，深度与视差、焦距、基线存在几何约束；模型学习利用双目证据而不是仅凭单图语义猜尺度。Stereo3D-Bench 覆盖点级与物体级共八项任务，并用右视图干预检验真实双目贡献。
- 项目页报告 StereoVLM-8B 深度 δ₁ 95.17%、物体级均分 66.98。匹配的 geometry-only 训练中，未见场景的视差监督深度 δ₁ 74.92%，直接深度监督 45.89%。五个空间基准平均 +3.54 分；相同 GR00T 架构及双目输入设置下 RoboCasa 成功率 38.1%→42.4%。
- 依赖标定和双目观测，错误外参、遮挡、纹理不足等需单独检查；RoboCasa 为仿真，不是实机泛化证明。代码与权重未确认发布，暂不能完整复现。

## 原文通读补充（2026-11-24）

### 方法与训练

- 整流双目满足 $d=u_l-u_r$、$Z=f_xB/d$；问答视差采用图宽 1000 的归一化坐标，深度以左相机 z 轴米数表示。训练在 Qwen3-VL-Instruct 2B/8B 上全参数 SFT，3000 步、学习率 1e-5、batch 64；15% 点级、10% 物体级双目问答和 75% 通用数据共同训练，而非依序训练。
- Stereo3D-Bench 包含 2017 几何图像对、1556 语义图像对、94546 问答；两类图像对独立选取，不能相加作为去重图像数。附录按 DROID 场景、Xperience-10M UUID、FSD 帧及 RoboCasa 任务隔离训练与评测；FSD 不提供物理深度参考。

### 对照、消融与边界

- 8B 对应和视差 MAE 分别为 3.80、3.76（归一化像素坐标），深度 δ₁ 95.17%，物体级平均 66.98，目标定位 Joint@0.25 为 51.28。geometry-only 的 OOD 视差推导深度 δ₁ 74.92%，直接深度 45.89%；完整模型的 OOD 指标为 68.92%/70.16%，不可混用两组实验。
- 2B 正确双目深度 δ₁ 94.51%，复制左图作为右图降至 7.4%；保留标定的单左图为 80.82%。五个迁移基准平均 47.94→51.48；RoboCasa 对照使用 24 任务、每任务 50 演示、60000 步训练、50 次评测 rollout，GR00T 同双目输入下 38.1→42.4%，不代表真机提升。

## 公开资源

- [论文 HTML](https://arxiv.org/html/2611.27915)
- [arXiv](https://arxiv.org/abs/2611.27915)
- [官方项目页](https://xpeng-robotics.github.io/stereovlm/)

## 对 wiki 的映射

- [stereovlm 独立实体](../../wiki/entities/paper-stereovlm.md)
- [关联归档](../sites/stereovlm.md)
