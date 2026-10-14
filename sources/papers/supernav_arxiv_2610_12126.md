# SuperNav: An Agentic Navigation System for Any Task in Any Scene

> 来源归档（ingest）— 根据 arXiv v1 论文与作者官方项目资料编译，不是原文镜像。

- **标题：** SuperNav: An Agentic Navigation System for Any Task in Any Scene
- **类型：** paper / agentic navigation / VLN / ObjectNav
- **作者：** Jinkai Zhang、Jingyi Xu、Yuanhong Yu、Jiarui Guo、Ruizhen Hu、Hujun Bao、Xiaowei Zhou、Sida Peng
- **机构：** 浙江大学；深圳大学；Causa Robotics
- **arXiv：** <https://arxiv.org/abs/2610.12126>（[HTML v1](https://arxiv.org/html/2610.12126v1)）
- **项目页：** <https://zju3dv.github.io/SuperNav/>
- **代码：** <https://github.com/zju3dv/SuperNav>
- **论文日期：** 2026-10-08（v1）
- **一句话说明：** 给预训练多模态大模型配上导航 agent harness，以视觉点目标、可调用导航工具、Skills 和任务进度管理支持跨任务/场景导航，不对 MLLM 做导航专用微调。

## 论文要点

SuperNav 将请求理解、场景判断与高层动作选择交给预训练 MLLM，由 harness 维持持续运行所需的状态，并通过导航工具执行观测与移动。核心视觉点接口让模型在四向 RGB 观测图中指定归一化像素坐标，执行器返回新观测与执行反馈，模型据此继续决策。

Harness 包括三个相互配合的部分：

1. **Agent loop：** 持续处理用户请求、观测、工具返回与交互历史；追踪待完成/已找到目标，支持多目标任务和显式结束。
2. **Agent-oriented tools：** 提供初始化、观测、转向、视觉点导航、目标进度更新与结束操作。任务完成声明由 MLLM 报告，基准另行核验成功。
3. **Navigation Skills：** 按需读取的 Markdown 指南，覆盖搜索、核验、受阻恢复等流程；Skills 提供指导，不会强制动作序列或代替模型判断。

视觉点导航可连接不同运动后端：几何执行器以深度、相机标定和位姿将像素点转到可达地图位置后规划路径；学习执行器根据图像目标点与近期 RGB 历史预测局部位移增量和停止信号。该接口把高层目的地选择与具体运动策略解耦。

## 评测摘要

| 场景 | SuperNav 结果 | 评测口径 |
|------|---------------|----------|
| 单物体导航 | SR 78.00%，SPL 0.4127 | Habitat-GS，150 项任务 |
| 有序多物体导航 | SR 34.00%，SPL 0.1388 | Habitat-GS，150 项任务 |
| 需求驱动导航 | SR 59.50%，SPL 0.1801 | AI2-THOR，200 项任务 |
| 类别导航 | SR 68.33% / 73.33%，SPL 0.3837 / 0.4105 | HM3D-OVON 未见验证集，分别为 0.25 m / 1 m 阈值 |
| 类别导航 | SR 80.30% / 86.50%，SPL 0.3338 / 0.3638 | HM3Dv2 验证集，分别为 0.2 m / 1 m 阈值 |

这些数据来自不同基准、阈值和协议，不能将各行 SR/SPL 直接横向比较。项目页还展示了 Unitree Go2 真机导航视频；论文与仓库说明要区分演示和可复现部署支持：截至归档时，公开仓库提供 agent runtime、仿真工具/评测和学习执行器权重，但暂不包含真机驱动。

## 对 wiki 的映射

- [SuperNav 详情页](../../wiki/entities/paper-supernav-agentic-navigation.md)
- [视觉–语言导航任务页](../../wiki/tasks/vision-language-navigation.md)
