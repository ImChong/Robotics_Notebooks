# ORA: Open Robot-Use Agents

> 来源归档（官方博客）

- **标题：** ORA: Open Robot-Use Agents
- **类型：** blog（Hao AI Lab 官方博客）
- **作者 / 发布方：** ORA Team；Hao AI Lab @ UC San Diego
- **原始链接：** <https://haoailab.com/blogs/ora/>
- **发布日期：** 2026-10-15
- **入库日期：** 2026-10-19
- **代码：** <https://github.com/hao-ai-lab/ORA>
- **一句话说明：** ORA 将机器人操作做成 ReAct 工具调用闭环：开放权重 LLM 根据 RGB-D 与机器人状态调用感知、测量、抓取和运动工具，每个动作经 supervisor 执行并返回回执；作者报告 RoboTwin 与 LIBERO-Pro 零样本仿真结果，同时说明成本、延迟和真机验证仍有限。

## 摘录要点（归纳）

- **方法定位：** robot-use agent。不同于 VLA 每步输出动作，或 LLM 先写开环程序，ORA 在每个有界动作后观察实测结果，再决定下一步。
- **任务上下文：** prompt bundle 包含系统策略、证据规则、基准边界、机器人卡片、任务说明与工具 schema；机器人卡片给出坐标系、控制点、夹爪几何和相机配置。
- **工具与边界：** 提供对象定位、分割、像素到世界坐标标记、几何查询、抓取规划、位姿移动及夹爪工具。supervisor 检查动作限制并执行，模型不能绕过它直接改环境。
- **证据管理：** 每轮输入当前观测和动作回执；旧结果保留引用供回看，降低重复图像占用并维持轨迹证据。
- **仿真结果（作者报告）：** DeepSeek-V4.1-Flash + ORA 在 RoboTwin C2R 250 个 case 中成功 173 个（69.2%），LIBERO-Pro 800 个 case 中成功 651 个（81.4%）。无有效结局的 case 计为失败。
- **限制：** 每次工具调用都需模型决策，带来延迟与 API 成本；成本估算不含 GPU 和仿真器。量化主结果来自仿真，真实机器人内容为初步演示。跨论文基线设置和报告 case 数未必完全一致。

## 项目页与源码开放核查

官方页面链接到 <https://github.com/hao-ai-lab/ORA>。仓库已公开，Apache-2.0（改编第三方代码保留其原许可）。仓库含 agent harness、机器人工具与 supervisor、LIBERO / LIBERO-Pro、RoboTwin 2.0、RoboDojo 入口和 terminal UI。完整 benchmark 流程建议 Linux + NVIDIA GPU（单 GPU 指引建议 32 GB 显存或以上）、uv 与模型 API key；真机演示不能等同于完整硬件部署复现。

## 对 wiki 的映射

- [ORA: Open Robot-Use Agents](../../wiki/entities/ora-open-robot-use-agents.md)
