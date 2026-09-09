# openvinotoolkit/physicalai — OpenVINO Physical AI 运行时

> 来源归档（以 GitHub README 与 docs.openvino.ai/2026/physical-ai 叙述为准）

- **标题：** OpenVINO™ Physical AI Runtime
- **类型：** repo / robotics / vla-deployment
- **仓库：** https://github.com/openvinotoolkit/physicalai
- **文档：** https://docs.openvino.ai/2026/physical-ai.html
- **训练侧配套：** [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio)（GUI 训练与导出）
- **入库日期：** 2026-09-09
- **开源状态：** **已开源**（Python 运行时；CLI 为 preview/planned API）
- **一句话说明：** OpenVINO **机器人策略部署运行时**：统一 **相机 API**（UVC / RealSense / Basler / IP）、**机器人协议**（结构类型）、**推理引擎**（加载 Studio 导出策略、自动选后端）与 **RobotRuntime 控制环**（观测构建 → 推理 → 动作下发）；支持 **PolicySource**（策略）与 **TeleopSource**（遥操作）等可插拔动作源；与 **LeRobot** 导出链对齐。
- **沉淀到 wiki：** [OpenVINO](../../wiki/entities/openvino.md)

---

## README 要点

### 定位

「Runtime package for deploying robot policies trained with Physical AI Studio」—— 在真机上运行训练好的策略：相机采集、机器人控制、策略推理 **统一 API**，跨硬件厂商。

### 核心模块

| 模块 | 能力 |
|------|------|
| **Unified Camera API** | UVC、RealSense、Basler、IP 相机同一接口 |
| **Robot Protocol** | `connect` / `disconnect` / `get_observation` / `send_action` + `joint_names`；无需继承基类 |
| **Inference Engine** | 加载 Studio 导出包；自动检测推理后端 |
| **Robot Runtime** | 控制环：硬件连接、读相机、建 observation、推理、dispatch action |
| **Action Sources** | `PolicySource`（策略模型）、`TeleopSource`（主从遥操作）、自定义 `ActionSource` |

### 部署工作流

```
exported policy package
    → InferenceModel
    → PolicyRuntime
    → Robot (+ cameras)
```

### 文档结构（仓库 `docs/`）

```
docs/
├── getting-started/   # 教程
├── how-to/            # 任务指南
├── explanation/       # 概念与边界（含 robots 协议）
└── reference/         # CLI、schema、API
```

### 与 LeRobot / Intel 栈

- Intel 产品页：**LeRobot 集成** — 导出 LeRobot 模型，PyTorch 或 OpenVINO 推理
- **Physical AI Studio**：Intel Robotics AI Suite 组件，预验证 VLA（ACT、SmolVLA、PI0.5 等）

---

## 对 wiki 的映射

- 写入 **`wiki/entities/openvino.md`** 的 Physical AI 与源码运行时序图
- 在 **`wiki/entities/lerobot.md`** 补充 Intel 部署路径一句

---

## 外部参考

- [openvinotoolkit/physicalai](https://github.com/openvinotoolkit/physicalai)
- [Physical AI 文档](https://docs.openvino.ai/2026/physical-ai.html)
- [Robots 协议说明](https://docs.openvino.ai/2026/physical-ai/explanation/robots.html)
