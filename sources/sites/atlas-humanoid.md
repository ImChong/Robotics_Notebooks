# ATLAS：Agentic Harness for Whole-Body Intelligence from Human Video（项目主页）

- **标题：** ATLAS: Agentic Harness for Whole-Body Intelligence from Human Video
- **类型：** site / official project page
- **URL：** <https://atlas-humanoid.github.io/>
- **技术原文：** <https://atlas-humanoid.github.io/blog/>（官方长文，Early release，2026）
- **机构：** UC Berkeley、NVIDIA Autonomous System & Physical AI Research Group、香港大学；作者单位还包括 Stanford
- **入库日期：** 2026-11-13
- **复核日期：** 2026-11-14
- **代码：** [atlas-humanoid/atlas-humanoid.github.io](https://github.com/atlas-humanoid/atlas-humanoid.github.io) 是**网站源代码**，含 HTML/CSS/JS、Remotion 展示动画和视频渲染工具；截至 2026-11-14 **未见 ATLAS 在线 RL、ScaleBFM 控制/真机部署实现或权重**（[仓库核查归档](../repos/atlas-humanoid-website.md)）。
- **论文：** 项目博客仅给出 2026 Early release 的 BibTeX；未列论文 PDF、arXiv 或 DOI（不得补造编号）
- **数据：** 项目页未提供独立数据集下载入口

## 一句话摘要

ATLAS 用单段人类示范视频和指令，由 VLM Agent 写任务规格、冻结的 ScaleBFM 跟踪全身动作，再由在线模型式强化学习修正末端指令，遇到失败自动重试；不依赖任务专用机器人示范或任务专用仿真训练。

## 项目页核验

- 官方自述的层级为 System 2（任务推理 VLM）↔ agentic harness（慢侧文本更新、快侧残差在线学习）↔ System 0（冻结全身 tracker），**不是**“完全没有快速学习器”；没有预训练的 System 1/VLA/WAM。
- ScaleBFM 运行 50 Hz；SAM 3 + 机载 RGB-D 估计目标，fast actor 在机器人上推理，慢侧 VLM 和训练更新在外部 PC 上运行。
- 项目页报告垃圾拾取任务一个连续 session 中 38 次尝试、最近八次成功率 0%→88%；这些是项目自报而非外部独立复现。
- 官网首页强调 all sensing onboard，但细节 FAQ 说明部分平地任务使用外部定位系统；对“完全仅机载感知”的适用范围不能泛化。
- 官方开放了网页和视频可视化代码，但未提供 ATLAS 算法/真机运行入口，故**ATLAS 源码运行时序图不适用**；只绘制依据原文的架构/控制流。

## SuperMimic 与在线 BPTT 的联系（2026-11-14 复核）

- 官方博客 §2.3 用真机尝试拟合局部末端命令–响应模型 `fψ`，经模型内 rollout 反向传播更新末端残差 `πφ`，参考文献列有 SuperMimic。两者共享可微模型优化策略的方法脉络。
- **不同点：** ATLAS 用冻结的 ScaleBFM 执行全身动作，只在线适应末端残差；SuperMimic 主要通过仿真数据训练全身 motion tracker。未发现 ATLAS 复用 SuperMimic 实现源码的公开依据。

## 对 wiki 的映射

- [ATLAS 项目实体](../../wiki/entities/atlas-agentic-harness.md)
- [官方技术长文归档](../blogs/atlas_humanoid_early_release.md)
