# AgiBot Teleoperation 项目页：真机遥操作与数据采集

- **类型：** 官方项目/系统介绍页
- **URL：** https://teleop.agibot.com/
- **关联论文：** https://arxiv.org/abs/2611.22388
- **核对日期：** 2026-11-19
- **核对范围：** 阅读 Scalability、Use Cases、Capabilities、AI Copilot、Interface、System、Solutions 与 Footer 展示的资源链接；核实项目与论文关系和代码/数据是否提供公开下载。
- **通读：** 已完成
- **论文归档：** [Hidden Variables Behind Real-Robot Data Quality](../papers/hidden-variables-real-robot-data-quality-arxiv-2611-22388.md)
- **wiki：** [独立论文详情](../../wiki/entities/paper-hidden-variables-real-robot-data-quality-2611-22388.md)

## 官网披露与证据边界

- 官方网页显示 **3000+ 部署机器人、5000+ 任务、200k 采集小时**，并展示跨城市遥操作与零售、厨房、家庭、工厂等场景。属于官网运营描述，不能据此推导为公开共享数据。
- 展示超视距通信（BLOS）、全身柔顺、全身控制、辅助驾驶、示教等能力，以及人机共驾、半自主采集和 1 对 N 监管的产品方向。
- 系统介绍把 human interface、网络视频回传、命令仲裁、Ego/UMI 重定向、全身柔顺控制、机器人运行时、视频/遥测/动作同步记录组织成闭环；这些是**网页系统层描述**，不等价于已经发布的论文复现代码。

## 代码与数据开放核查

- **本论文对应代码：** 截至 2026-11-19，官网未列出可核验的本论文实验、训练或推理代码仓库入口；论文 HTML 亦未提供实验复现仓库。
- **本论文数据集：** 未发现可直接下载论文所用 207,030 小时示教库或实验原始 episode 的入口。
- 官网另链 RoboClaw、Semi-auto Data Collection、1-to-N supervision 等**其他独立项目**；不能把这些外链误判为本论文官方代码。
- **开放状态：** 本论文复现代码与论文数据集**尚未发现公开入口**。不创建伪造的 sources/repos/ 记录，不绘制虚构的源码运行时序图。
