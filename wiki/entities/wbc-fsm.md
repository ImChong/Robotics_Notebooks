---
type: entity
tags: [repo, wbc, fsm, humanoid, unitree-g1, deployment, onnx, cpp, motion-tracking, whole-body-control]
status: complete
updated: 2026-05-01
related:
  - ./amp-mjlab.md
  - ./unitree-g1.md
  - ./unitree-rl-mjlab.md
  - ../concepts/whole-body-control.md
  - ../methods/motion-retargeting-gmr.md
  - ../tasks/locomotion.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/repos/wbc_fsm.md
summary: "wbc_fsm 是针对 Unitree G1 的 C++ 全身控制部署框架，用有限状态机管理 Passive/Loco/WBC 三种模式，内嵌 LAFAN1 动捕训练的 ONNX 策略，无 ROS 依赖，支持仿真与真机双端部署。"
---

# wbc_fsm (G1 全身控制 FSM 部署框架)

**wbc_fsm** 是 **ccrpRepo / ZSTU Robotics** 针对 **Unitree G1** 人形机器人开发的 C++ 部署框架，以**有限状态机（FSM）**组织多种控制模式，核心是一个基于 LAFAN1 动捕数据训练的 ONNX 全身运动追踪策略。

## 一句话定义

用有限状态机统一管理人形机器人的安全保护、行走、全身动作追踪三种模式，内置预训练 ONNX 策略，C++ 原生部署，无 ROS 依赖。

## 为什么重要

- **FSM 作为运动控制运行时架构**：多种控制模式（安全 / 行走 / 全身控制）在真机上的组织方式是工程实践的重要范式，wbc_fsm 提供了一个干净的 C++ 参考实现
- **ONNX C++ 推理闭环**：展示从训练（Python/ONNX 导出）→ 真机推理（C++ ONNX Runtime）的完整 Sim2Real 链路
- **与 AMP_mjlab 形成配对**：同属 ccrpRepo，AMP_mjlab 负责策略训练（Python + mjlab），wbc_fsm 负责策略部署（C++ + Unitree SDK2）

## FSM 状态机设计

wbc_fsm 的核心是三态状态机：

```
上电
 ↓
[Passive 阻尼保护]  ← 默认安全态，关节阻尼制动
 ↓ START
[Loco 行走模式]     ← locomotion 控制
 ↓ R1+Up
[WBC 全身控制]      ← ONNX 策略驱动，跟踪 LAFAN1 动捕参考
```

**设计原则**：
- 从安全态 → 运动态单向升级，不跳跃；
- 每个状态独立 JSON 配置，解耦参数；
- 手柄触发模式切换，保证操作员可随时中止。

## 主要技术路线

**路线：MoCap 重定向 → RL 训练 → ONNX 导出 → C++ 部署**

1. **动捕数据**：LAFAN1 MoCap 数据集 → 重定向到 G1 骨架关节空间
2. **策略训练**：RL（AMP 风格）学习跟踪动捕参考动作（训练侧由 AMP_mjlab 等框架完成）
3. **ONNX 导出**：策略网络序列化为 `lafan1_0128_1.onnx`
4. **C++ 推理**：ONNX Runtime 1.22.0 在 G1 PC2（aarch64）上实时推理，驱动全身关节

## 部署架构

```
G1 板端 (PC2 / aarch64)
├── wbc_fsm 主进程
│   ├── FSM 控制器          ← 状态转换逻辑
│   ├── ONNX Runtime        ← lafan1_0128_1.onnx 策略推理
│   ├── 运动参考播放器       ← motion_data/lafan1/ 参考序列
│   └── Unitree SDK2 接口   ← 关节指令下发 / 传感器读取
└── 手柄输入               ← 模式切换触发
```

**关键参数**：
- 推理引擎：ONNX Runtime 1.22.0（aarch64 / x64 双版本）
- 通信接口：真机 `eth0` / 仿真 `lo`
- 构建：CMake ≥ 3.14，C++ 17，`-O3 -pthread`

## 与 AMP_mjlab 的互补关系

| 维度 | AMP_mjlab | wbc_fsm |
|------|-----------|---------|
| 语言 | Python | C++ |
| 阶段 | 策略训练（仿真） | 策略部署（真机） |
| 框架 | mjlab + rsl_rl | Unitree SDK2 |
| 输出 | ONNX 模型文件 | 关节指令流 |
| 同组织 | ✅ ccrpRepo | ✅ ccrpRepo |

## 局限性

- **仅部署，无训练代码**：策略改进需依赖外部训练框架（如 AMP_mjlab）
- **硬绑 G1 + Unitree SDK2**：移植到其他机器人平台需大幅重写接口层
- **动作类型受限**：当前 ONNX 模型仅覆盖 LAFAN1 dance12 动作，扩展需重新训练

## 参考来源

- [sources/repos/wbc_fsm.md](../../sources/repos/wbc_fsm.md) — 仓库原始归档
- [ccrpRepo/wbc_fsm GitHub](https://github.com/ccrpRepo/wbc_fsm) — 代码仓库
- Harvey et al., *Robust Motion In-Betweening*, SIGGRAPH 2020 — LAFAN1 数据集来源论文

## 关联页面

- [AMP_mjlab](./amp-mjlab.md) — 同组织，策略训练侧对应物
- [Unitree G1](./unitree-g1.md) — 目标硬件平台
- [unitree_rl_mjlab](./unitree-rl-mjlab.md) — Unitree 官方 RL+ONNX 部署参考
- [Whole-Body Control](../concepts/whole-body-control.md) — WBC 理论概念页
- [Motion Retargeting GMR](../methods/motion-retargeting-gmr.md) — LAFAN1 → G1 重定向方法
- [Locomotion](../tasks/locomotion.md) — 上层任务场景
