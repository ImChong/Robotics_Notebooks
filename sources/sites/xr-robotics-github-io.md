# XRoboToolkit 项目页（xr-robotics.github.io）

> 来源归档

- **标题：** XRoboToolkit: A Cross-Platform Framework for Robot Teleoperation
- **类型：** site（项目页 + 视频 + BibTeX）
- **URL：** <https://xr-robotics.github.io/>
- **论文：** <https://arxiv.org/abs/2508.00097>
- **代码：** <https://github.com/XR-Robotics>（页头 GitHub 图标；**已开源**）
- **数据集：** 未在项目页列出独立公开数据集 URL（VLA 实验为自采 100 条折毯示范）
- **入库日期：** 2026-07-22
- **一句话说明：** ByteDance PICO 等团队的 XR 遥操作框架官方页：摘要、应用示意、补充视频与 BibTeX；标注 SII 2026 Best Paper；入口指向 XR-Robotics GitHub 组织。

## 开源核查（步骤 2.5）

| 项 | 结论（2026-07-22） |
|----|-------------------|
| 项目页 Code / GitHub | 页头有 GitHub 链 → `https://github.com/XR-Robotics` |
| 开放程度 | **已开源**（多仓：Unity Client、PC-Service、Sample、Vision、ROS 等） |
| 权重 / 数据集 | 项目页**未**挂 HF/Zenodo；π₀ 微调数据需自采 |
| 交叉归档 | 论文 [xrobotoolkit_arxiv_2508_00097.md](../papers/xrobotoolkit_arxiv_2508_00097.md)；代码 [xrobotoolkit.md](../repos/xrobotoolkit.md) |

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Hero / 奖项 | 标题 + **Best Paper Award, SII 2026**；作者与通讯（Ke Jing / Ning Yang） |
| Abstract | OpenXR 跨平台 XR 遥操作；低延迟立体反馈；QP-IK；多模态追踪；VLA 数据验证 |
| Video | 补充演示（与论文 YouTube 一致） |
| Applications | 双臂/移动操作、双 UR5+头跟踪、elbow tracker、MuJoCo 灵巧手 |
| BibTeX | `@article{zhao2025xrobotoolkit,... arXiv:2508.00097}` |

## 对 wiki 的映射

- 主实体：[XRoboToolkit（论文实体）](../../wiki/entities/paper-xrobotoolkit.md)
- 论文摘录：[xrobotoolkit_arxiv_2508_00097.md](../papers/xrobotoolkit_arxiv_2508_00097.md)
- 代码仓库：[xrobotoolkit.md](../repos/xrobotoolkit.md)
- 任务页：[teleoperation.md](../../wiki/tasks/teleoperation.md)
