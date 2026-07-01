# Tnkr — Open Duck Mini V2 项目文档

> 来源归档

- **标题：** Open Duck Mini / Open Duck Mini V2（Tnkr 项目页）
- **类型：** site（协作平台项目文档 + 分步装配/线束/软件指南）
- **来源：** Tnkr / Open Duck Project（apirrone）
- **链接：** https://tnkr.ai/open-duck-mini/open-duck-mini-v2
- **旧路径（仍可能跳转）：** https://tnkr.ai/explore/docs/open-duck-mini/open-duck-mini-v2
- **入库日期：** 2026-07-01
- **一句话说明：** Tnkr 上 Open Duck Mini v2 的**官方推荐组装入口**：结构化文档树覆盖打印/BOM/分步装配/线束/Runtime 部署与策略训练，并托管 STL 下载与社区 Pull Requests。
- **沉淀到 wiki：** 交叉更新 [`wiki/entities/open-duck-mini.md`](../../wiki/entities/open-duck-mini.md)、[`wiki/entities/tnkr.md`](../../wiki/entities/tnkr.md)、[`wiki/entities/open-duck-mini-runtime.md`](../../wiki/entities/open-duck-mini-runtime.md)

---

## 为什么值得保留

- GitHub Hub 仓 [`assembly_guide.md`](../repos/open_duck_mini.md) 仍标注 incomplete；README 明确指向 **Tnkr 指南** 作为 v2 主线装配文档。
- 相比纯 Markdown，Tnkr 页把 **Hardware（打印→BOM→装配→线束）** 与 **Software（Runtime→部署→训练）** 收在同一项目空间，并带 **Files / Pull Requests** 协作入口，是「整机复现」的一手导航。
- 页面为 Next.js SPA；自动化抓取需解析 RSC 载荷或浏览器渲染（2026-07 入库时采用 RSC 解析 + GitHub 仓交叉核对）。

## 项目概览（页面 Home）

| 字段 | 内容 |
|------|------|
| 名称 | Open Duck Mini V2 |
| 简介 | Making a mini version of the BDX droid. |
| 平台 Tab | Home · Documentation · Files · Pull Requests · Help Center |
| 商业套件 | 页面推广 **Open Duck kit box**（frame + electronics + assembly hardware 一站式采购，链接以站点为准） |

## 文档树（Documentation）

### Introduction

| 节点 | 类型 | 说明 |
|------|------|------|
| Overview | documentation | 项目总览 |

### Hardware

| 顺序 | 节点 | 类型 | 要点 |
|------|------|------|------|
| 1 | Print Guide | documentation | 除 `foot_bottom_tpu.stl` 外 **PLA 15% infill**；TPU 件 **40% infill** |
| 2 | Bill of Materials | bom | 物料表（与 Google Sheets / GitHub BOM 对齐） |
| 3 | Assembly Instructions | assembly | **交互式分步装配**（见下节） |
| 4 | Electronics & Wiring | documentation | 全局线束示意、Servo Connection Map、Pi Zero 2W 供电与 IMU/扬声器 |

### Software

| 顺序 | 节点 | 类型 | 要点 |
|------|------|------|------|
| — | Open Duck Mini Runtime | documentation | Pi OS Lite、venv、`Open_Duck_Mini_Runtime` v2 分支安装 |
| — | Deployments | 分组 | |
| — | Deploy your robot | documentation | 上机 checklist、偏置、手柄、ONNX 行走 |
| — | Train your policies | documentation | 链到 Playground 训练与 checkpoint → ONNX |

## 装配主线（Assembly Instructions 步骤标题）

**前置（必须在机械装配前完成）：**

1. **Configure Your Motors** — 逐台 Feetech 写 ID 并对零；使用 [`Open_Duck_Mini_Runtime`](https://github.com/apirrone/Open_Duck_Mini_Runtime) 的 `configure_motor.py --id <id>`；需独立供电（如电池组）。
2. **Apply Loctite Threadlocker** — 金属对金属螺钉用 **Loctite 243**；**塑料螺钉禁用**。

**机械子装配顺序（与 GitHub `assembly_guide.md` 一致，Tnkr 为分步可视化版）：**

| 阶段 | 关键步骤 |
|------|----------|
| Trunk | Insert Bearings and M3 Inserts → Assemble Trunk Parts → Mount Neck Pitch Motor → Add Roll Motor Bottom |
| Feet | TPU/PLA 脚底粘合 → M3 热熔螺母 → 装 foot motor（driver 侧朝向 `foot_top`）→ Foot Switches |
| Shins | Leg spacer 热熔螺母 → 穿线 → Attach Knee Motors |
| Thighs | Assemble Thigh Parts → Mount Hip Pitch Motor（**零位方向关键**） |
| Hips | Mount Roll Pitch → Roll Motor → Hip Roll → 与 trunk 合体 → Attach Leg Subassembly |
| Neck / Head | Mount Head Pitch Motor → Head Mechanism |
| Body / Electronics | IMU、电池组、Pi Zero 2W、扬声器、全局线束 |

**通用材料：** 烙铁与基础电工工具、M3 螺钉（GitHub 指南 TODO 精确数量）、Loctite 243、线材；随时参照 [Onshape CAD v2](https://cad.onshape.com/documents/64074dfcfa379b37d8a47762/w/3650ab4221e215a4f65eb7fe/e/0505c262d882183a25049d05)。

## 14 路舵机 ID 映射（Runtime / Tnkr 一致）

| 关节 | ID | 关节 | ID |
|------|-----|------|-----|
| right_hip_yaw | 10 | left_hip_yaw | 20 |
| right_hip_roll | 11 | left_hip_roll | 21 |
| right_hip_pitch | 12 | left_hip_pitch | 22 |
| right_knee | 13 | left_knee | 23 |
| right_ankle | 14 | left_ankle | 24 |
| — | — | neck_pitch | 30 |
| — | — | head_pitch | 31 |
| — | — | head_yaw | 32 |
| — | — | head_roll | 33 |

## Software 摘录要点

### Raspberry Pi Zero 2W

- **系统：** Raspberry Pi OS Lite（64-bit）；Imager 预配 SSH / WiFi（文档建议手机热点便于远程）。
- **Runtime 安装：**
  ```bash
  git clone https://github.com/apirrone/Open_Duck_Mini_Runtime
  cd Open_Duck_Mini_Runtime && git checkout v2 && pip install -e .
  ```
- **Pi 5 注意：** 需 `pip uninstall -y RPi.GPIO && pip install lgpio`。
- **电机板 udev：** 设置 FTDI `latency_timer=1`（低延迟 USB）。
- **验证：** `check_motors.py`、IMU `raw_imu.py` / `imu_server.py`、Xbox 蓝牙 `xbox_controller.py`。

### 部署与预演

- 关节软偏置写入配置文件（未来计划写入 EEPROM）。
- MuJoCo 预演：`python v2_rl_walk_mujoco.py --onnx_model_path <path>/BEST_WALK_ONNX_2.onnx`
- 预训练 ONNX：[Hub 仓 v2](https://github.com/apirrone/Open_Duck_Mini/blob/v2/BEST_WALK_ONNX_2.onnx)

### 训练

- 链到 [Open_Duck_Playground](../repos/open_duck_playground.md)：MJX 训练 → 导出 ONNX → Runtime 上机。

## Files 与 STL 托管

- 打印件 STL 经 **tnkr-cdn.s3.eu-west-2.amazonaws.com** 分发（如 `roll_motor_top.stl`、`foot_bottom_tpu.stl`、`trunk_top.stl` 等）；与 Onshape 导出 / GitHub 发布需以**当前 Tnkr Files 页**为准。
- BOM 页内表格可直链各零件 STL。

## 与 GitHub 文档分工

| 资料 | 角色 |
|------|------|
| **本 Tnkr 页** | v2 **主线**装配/线束/Runtime 导航；交互步骤 + 社区 PR |
| [`Open_Duck_Mini` v2 docs](../repos/open_duck_mini.md) | CAD/BOM/sim2real 原理、预训练 ONNX、`print_guide.md` |
| [`Open_Duck_Mini_Runtime` v2](../repos/open_duck_mini_runtime.md) | 机载代码与 checklist 源码 |

## 对 wiki 的映射

| 主题 | wiki 页 |
|------|---------|
| 整机与 Tnkr 入口 | `wiki/entities/open-duck-mini.md` |
| Tnkr 平台 + 范例项目 | `wiki/entities/tnkr.md` |
| 机载 Runtime / 电机 ID | `wiki/entities/open-duck-mini-runtime.md` |
| Sim2Real 概念 | `wiki/concepts/sim2real.md` |
