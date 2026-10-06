# Open GRID 与 GRID 文档（General Robotics）

> 来源归档

- **标题：** Open GRID / GRID Docs v2.1
- **类型：** site（开放 Web 平台 + 开发者文档）
- **URL：**
  - Open GRID：<https://grid.generalrobotics.dev>
  - 文档索引：<https://docs.generalrobotics.dev/llms.txt>
  - 入门：<https://docs.generalrobotics.dev/v2.1/introduction.md>
  - 安装 CLI：<https://docs.generalrobotics.dev/v2.1/get-started/installation.md>
- **入库日期：** 2026-09-27
- **一句话说明：** **Open GRID** 为浏览器/Web 入口（README 称免安装）；**v2.1 文档** 描述 GRID CLI、云 Isaac Sim / AirGen 会话、真机 VS Code 工作区与 **GRID Cortex** 托管模型 API（`grid_cortex_client`）。

## 核查说明

- 用户提供的旧路径 `docs.generalrobotics.dev/grid/open-grid/introduction` 返回 **404**；当前文档根为 **v2.1**（见 `llms.txt`）。
- Open GRID 根域可能返回 API JSON（无浏览器路由时）；以 README 与产品页链接为准。

## 文档要点（策展）

- **CLI 一键安装** → 组织 **GRID cluster** 会话（真机可选，仿真可零硬件）。
- **仿真：** NVIDIA Isaac Sim 与 **AirGen** 会话配置（workflow / env / MDP / agent）。
- **Cortex：** `CortexClient().run(ModelType.OWLV2, ...)` 等统一 Python 调用检测/深度/分割/VLM/VLA。
- **部署：** 相机外参标定、Navigator iOS 支持页等。

## 对 wiki 的映射

- [GRID（General Robotics）](../../wiki/entities/paper-grid-general-robot-intelligence-development.md)
- [grid-playground.md](../repos/grid-playground.md)
