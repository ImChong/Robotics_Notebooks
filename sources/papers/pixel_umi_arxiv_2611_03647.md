# Pixel-UMI: Learning Environment-Aware Whole-Body Manipulation from Sensorized Human Demonstrations

- **中文标题：** Pixel-UMI：从传感化人类示范中学习环境感知全身操作
- **类型：** 论文，arXiv 预印本
- **作者：** Jason Jingzhou Liu、Tony Tao、Deepak Pathak、Ruslan Salakhutdinov
- **机构：** 卡内基梅隆大学（Carnegie Mellon University）
- **论文：** <https://arxiv.org/abs/2611.03647>（v1，2026-11-02）
- **项目页：** <https://pixel-robot.com/pixel-umi/>
- **相关系统论文：** [Pixel](pixel_arxiv_2611_03648.md)
- **入库日期：** 2026-11-04

## 初步摘要

论文提出从传感化人类示范学习移动机器人的全身操作：环境感知重定向利用重建的场景几何生成避碰姿态与轨迹，结合延迟校准的动作标注、基于重定向误差的视觉遮蔽，以及针对接触丰富任务的 DAgger 人工纠正。论文报告五项任务的平均任务进度为 87.0%；其中两项使用了机器人纠正数据，这一指标不是任务成功率。

## 待完成

- 两个项目页当前无法打开，代码、数据与权重的开放状态未核实。
- 论文全文尚未完成通读；需核实方法、实验和局限，并与 Pixel 系统论文一并整理为适当的项目实体，附基于原文的 Mermaid 流程图。
- 若存在可运行官方代码，阅读运行入口与相关源码并补充源码运行时序图。
- 尚未运行仓库要求的 `make ci-preflight`。此文件仅为资料占位，不能视为完整收录。
