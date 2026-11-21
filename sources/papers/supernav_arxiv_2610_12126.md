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
- **通读：** 已完成
- **一句话说明：** 通过 Codex CLI + MCP 工具（`hab_init_scene` / `hab_panorama` / `hab_turn` / `hab_visual_point_navigate` / `hab_close_session`）+ 按需 Markdown Skills + 仅保留关键视觉观测的上下文裁剪，把预训练冻结 MLLM（GPT-5.6 Terra / GPT-6 Astra）直接编排为通用导航 Agent；无需导航微调即可衔接可替换的几何执行器与基于 Flow Matching 的学习执行器。

## 原文通读要点（arXiv:2610.12126v1）

SuperNav 将请求理解、场景判断与高层动作选择交给冻结预训练 MLLM（实验主力 GPT-5.6 Terra 与 GPT-6 Astra），由基于 Codex CLI 与 MCP（Model Context Protocol）协议的 agent harness 维持长程循环、结构化任务进度与仅媒体保留的上下文裁剪。核心 4 视角 PointNav(`view`, `[u, v]`) 接口让模型在前后左右四向 RGB 中指点，执行器返回新观测与执行诊断。

Harness 与执行 backends 详情：

1. **Agent loop 与 MCP 工具链：** 提供 `hab_init_scene`、`hab_panorama`、`hab_turn`、`hab_visual_point_navigate` / `hab_local_navigate`、`hab_close_session` 等工具与进度管理；任务声明完成由 MLLM 触发但独立核验。
2. **Navigation Skills 与上下文裁剪：** 以按需读取的 Markdown 指南组织 Search / Recheck / Recover / Complete 等策略包，并在长程历史中进行 media-only 上下文裁剪防止 token 溢出。
3. **两类可互换运动后端：**
   - **Geo-based Executor：** 基于深度与 NavMesh（在 Go2 实机中采用 4 路 RGB + Hesai LiDAR 构筑体素地图与 `ReplanningAStar`）规划到达像素反投影目标点的路径。
   - **Learned Executor（Flow-based）：** 借鉴 NoMaD，采用 DINOv2 ViT-S/14 编码器 + 年龄感知（age-aware）≤12 帧历史 Transformer + 4 步 1D UNet Flow Matching 生成短时局部轨迹 chunk 与停止信号。

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
