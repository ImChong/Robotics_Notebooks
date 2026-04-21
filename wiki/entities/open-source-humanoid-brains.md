---
type: entity
tags: [humanoid, hardware, brain, computer, embedded, nvidia]
status: complete
updated: 2026-04-21
related:
  - ./humanoid-robot.md
  - ../queries/real-time-control-middleware-guide.md
  - ../roadmaps/humanoid-control-roadmap.md
sources:
  - ../../sources/papers/humanoid_hardware.md
summary: "主流开源人形机器人“大脑”（主控电脑）选型：对比了 NVIDIA Jetson Orin、高性能 X86 工控机及国产边缘算力平台的性能边界与适用场景。"
---

# 开源人形机器人“大脑” (主控电脑) 选型

对于人形机器人，其“大脑”需要承担两类截然不同的计算任务：一是需要极高确定性的底层 **运控循环 (1kHz+)**；二是需要海量算力的 **感知与大模型推理 (5-30Hz)**。

## 主流选型对比

| 类别 | 代表方案 | 优势 | 劣势 | 推荐场景 |
|------|---------|------|------|---------|
| **高性能 X86 工控机** | Intel NUC 13/14, 各种迷你 PC | CPU 单核主频极高，适合运行复杂的 WBC/MPC 优化 | 功耗较高 (60W-120W)，体积略大 | 纯运控、复杂动力学求解 |
| **NVIDIA Jetson** | Jetson Orin AGX / Nano | GPU 算力极其强大，原生支持 TensorRT 部署 VLA | 实时内核适配相对繁琐 (L4T) | 视觉感知、端到端学习部署 |
| **国产边缘算力平台** | 地平线旭日, 瑞芯微 RK3588 | 性价比极高，体积紧凑 | 算法生态相对封闭 | 简单 Locomotion、量产低成本方案 |

## 1. 为什么主频（Single-core Performance）对运控至关重要

足式机器人的 [WBC](../concepts/whole-body-control.md) 往往涉及大规模稀疏矩阵的 QR 分解。这类计算很难在 GPU 上并行，而是极度依赖 CPU 的单核性能。
- **推荐**：使用 i7-13700H 或更高级别的 X86 处理器作为运控核心。

## 2. 混合计算架构 (Hybrid Architecture)

目前最成熟的方案通常是“双大脑”或“分片大脑”：

- **大脑 (Vision/Language)**：跑在 NVIDIA Orin AGX 上，负责处理深度相机数据和 VLA 策略推理。
- **小脑 (Control/WBC)**：跑在高性能 X86 IPC 上，通过 [LCM/共享内存](../comparisons/ros2-vs-lcm.md) 接收大脑下发的 Action Chunk。

## 3. 选型 Checklist

- [ ] **物理接口**：是否具备双千兆网口（用于 EtherCAT 主站与感知相机分离）。
- [ ] **供电电压**：是否能直接支持 12V-48V 宽压输入，避免因电机启动压降导致重启。
- [ ] **实时补丁兼容性**：该主板是否能稳定运行带有 `PREEMPT_RT` 的 Linux 内核。
- [ ] **散热设计**：人形机器人机身封闭，主控是否带有强制主动散热方案。

## 关联页面
- [人形机器人 (Humanoid Robot)](./humanoid-robot.md)
- [实时运控中间件配置指南](../queries/real-time-control-middleware-guide.md)
- [Humanoid Control Roadmap](../roadmaps/humanoid-control-roadmap.md)

## 参考来源
- [humanoid_hardware.md](../../sources/papers/humanoid_hardware.md)
- 各主流机器人公司（如 Unitree, Digit）公开的技术规格书。
