---
type: entity
tags: [deployment, sim2real, embedded, raspberry-pi, open-source, biped]
status: complete
updated: 2026-05-28
related:
  - ./open-duck-mini.md
  - ./open-duck-playground.md
  - ../concepts/sim2real.md
  - ../concepts/processor-in-the-loop-sim2real.md
sources:
  - ../../sources/repos/open_duck_mini_runtime.md
summary: "Open Duck Mini Runtime 在 Raspberry Pi Zero 2W 上运行 ONNX 行走策略，负责 Feetech 舵机 I2C、IMU、Xbox 手柄与关节偏置标定，是 Open Duck sim2real 的机载最后一环。"
---

# Open Duck Mini Runtime

**Open Duck Mini Runtime** 负责在 **Raspberry Pi Zero 2W** 上将 [Open Duck Playground](./open-duck-playground.md) 导出的策略部署到真机：舵机总线、IMU 融合、手柄遥操作与上机前 checklist。

## 为什么重要

- **极低算力部署范例：** 在 Zero 2W 级 SBC 上跑 locomotion ONNX，对「策略频率 vs 控制频率分频」有现实约束。
- **硬件—软件对齐：** 关节软偏置标定（`find_soft_offsets.py`）直接写入驱动层，弥补廉价舵机零位不一致。
- **与 Hub 仓预训练策略衔接：** 可直接使用 `BEST_WALK_ONNX*.onnx` 或自训 ONNX，经 MuJoCo 脚本预演后再上机。

## 核心结构/机制

| 环节 | 说明 |
|------|------|
| OS | Raspberry Pi OS Lite 64-bit；SSH / WiFi 预配置 |
| 总线 | I2C 启用；Feetech PWM 控制（`hwi_feetech_pwm_control.py`） |
| 低延迟 USB | udev 将 FTDI `latency_timer` 设为 1 |
| 输入 | Xbox One 蓝牙配对；`test_xbox_controller.py` |
| 推理 | `v2_rl_walk_mujoco.py` + ONNX；部署 checklist |
| 标定 | `find_soft_offsets.py` → 各关节 `joints_offsets` |

## 常见误区或局限

- **checklist 不可跳过：** 电源、偏置、IMU 与通信未通过时不应直接加载行走策略。
- **README 仍标注 TODO：** 电机板 udev、Rust 绑定等条目在演进中；以 `v2` 分支 checklist 为准。
- **算力边界：** 复杂感知或高维策略需更强机载计算；Runtime 聚焦 locomotion 闭环。

## 参考来源

- [sources/repos/open_duck_mini_runtime.md](../../sources/repos/open_duck_mini_runtime.md)
- [apirrone/Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime)（v2 分支）

## 关联页面

- [Open Duck Mini](./open-duck-mini.md)
- [Open Duck Playground](./open-duck-playground.md)
- [Sim2Real](../concepts/sim2real.md)
- [Processor-in-the-Loop Sim2Real](../concepts/processor-in-the-loop-sim2real.md)

## 推荐继续阅读

- [Open Duck Mini sim2real 文档](https://github.com/apirrone/Open_Duck_Mini/blob/v2/docs/sim2real.md)
- [Runtime v2 checklist](https://github.com/apirrone/Open_Duck_Mini_Runtime/blob/v2/checklist.md)
