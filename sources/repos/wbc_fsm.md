# wbc_fsm

> 来源归档（ingest）

- **仓库：** https://github.com/ccrpRepo/wbc_fsm
- **类型：** repo（C++ 部署框架）
- **组织：** ccrpRepo / ZSTU Robotics
- **主语言：** C++ (98.8%)
- **Stars：** 138 | **Forks：** 23
- **入库日期：** 2026-05-01
- **沉淀到 wiki：** 是 → [`wiki/entities/wbc-fsm.md`](../../wiki/entities/wbc-fsm.md)
- **核对范围：** 完整阅读官方 `README.md` / `README_zh.md`；核查 2026-10-23 版本 `src/main.cpp`、`src/FSM/FSM.cpp`、`State_WBC.cpp` 观测/动作/状态切换、`State_Passive.cpp`、`State_FixedStand.cpp`、`State_Loco.cpp` / `State_Amp.cpp` / `State_MJAmp.cpp` 的转移条件、`src/interface/IOSDK.cpp`、`src/control/ControlFrame.cpp` 与 `config/wbc.json` / `config/loco.json`。
- **通读：** 已完成

## 一句话说明

针对 **Unitree G1 人形机器人** 的 C++ 全身控制部署框架，以有限状态机（FSM）组织 Passive / FixedStand / Loco / AMP / MJAMP / WBC 六个运行状态，内嵌基于 LAFAN1 动捕数据训练的 ONNX 策略，支持仿真（Unitree Mujoco）与真机（G1 PC2 / aarch64）部署。

## 为什么值得保留

1. **WBC+FSM 部署范式的具体实现**：演示了如何在真实人形机器人上用状态机组织多种控制模式，是 WBC 从算法到系统集成的典型参考。
2. **纯 C++ + Unitree SDK2 + ONNX Runtime**：无 ROS 依赖，轻量易移植，是 Sim2Real ONNX 部署模式的完整示例。
3. **来自 ccrpRepo**：与 AMP_mjlab 同一组织出品，可与 AMP_mjlab 的训练侧形成"训练（Python/mjlab）→部署（C++/FSM）"闭环参考。
4. **LAFAN1 动捕参考驱动**：展示了将 MoCap 数据提炼为 ONNX 策略再上机的完整链路。

## 核心技术摘录

### FSM 状态机结构（以 4b9afa27 源码为准）

`FSM.cpp` 注册六态：**Passive / FixedStand / Loco / AMP / MJAMP / WBC**。主路径为 `Passive --START--> FixedStand --R2+A--> MJAMP --R1+Up--> WBC`；运动状态之间还可按手柄指令互切，`L2+B` 可回退 Passive，WBC 姿态偏差超阈值也触发回退。此前“三态”仅是 README 对主要功能的概括，并非源码注册状态总数。

### 运动模型

- **策略**：`model/wbc/dance12_0207_1.onnx`（当前 `config/wbc.json`，ONNX Runtime 1.22.0）
- **运动参考**：`motion_data/lafan1/dance12`（当前 `config/wbc.json` 的 `motion_path`）
- **推理引擎**：ONNX Runtime，x64（仿真）/ aarch64（真机 G1 PC2）

### 目录结构

```
wbc_fsm/
├── config/              # JSON 配置（wbc / loco / fixedpose / passive）
├── include/
│   ├── FSM/             # 状态机接口
│   ├── control/         # 控制组件
│   ├── interface/       # 硬件通信接口
│   └── message/         # 消息定义
├── src/                 # 各模块实现
├── model/wbc/           # 预训练 ONNX 策略
└── motion_data/lafan1/  # MoCap 参考动作
```

### 依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| Unitree SDK2 | — | 机器人通信层 |
| ONNX Runtime | **1.22.0** | 策略推理 |
| Eigen3 | — | 线性代数 |
| nlohmann_json | ≥ 3.7.3 | JSON 配置 |
| CMake | ≥ 3.14 | 构建 |
| C++ | 17 | 语言标准 |

### 构建与部署

```bash
# 编译
mkdir -p build && cd build && cmake .. && make -j4

# 仿真：先启动 Unitree Mujoco，接口设为 lo
# 真机：复制到 G1 PC2 /home/unitree，接口设为 eth0
```

### 手柄操作（源码 `checkChange`）

| 按键 | 源码行为 |
|------|----------|
| START | Passive → FixedStand |
| R2 + A | FixedStand / Loco → MJAMP（WBC → AMP） |
| R2 + B | AMP / MJAMP → Loco |
| R1 + Up | Loco / AMP / MJAMP → WBC |
| L2 + B | 运动状态 → Passive |
| WBC 内 R2 / L2 | 暂停到预设帧 / 当前帧 |
| WBC 内 R1 | 恢复动作 |
| SELECT | 抛异常以请求程序退出 |

## 与其他 ccrpRepo 项目的关系

```
AMP_mjlab (Python / mjlab)    →    独立训练侧示例
                                         ↓
              wbc_fsm (C++ / Unitree SDK2)  →  部署上机（G1 PC2）
```

wbc_fsm 是 AMP_mjlab 系列的**部署侧对应物**：AMP_mjlab 负责策略训练（Python/IsaacLab API），wbc_fsm 负责策略执行（C++/真机）。

## 当前提炼状态

- [x] 仓库定位与 FSM 状态机结构
- [x] ONNX 部署链路（训练 → ONNX → C++ 推理）
- [x] 与 AMP_mjlab / WBC 体系的关联关系
- [ ] 后续可深挖：config JSON 参数解读、FSM 转移条件形式化、与 unitree_rl_mjlab 部署模式对比

## 2026-11-21 官方原文复核与差异

- **版本**：[ccrpRepo/wbc_fsm@4b9afa27](https://github.com/ccrpRepo/wbc_fsm/commit/4b9afa27ad4eeab8e95584c67c91275b9831fc4d)（2026-10-23）。官方 README 已完整阅读，关键源码路径按上方「核对范围」检查。
- **文档与配置不一致**：README 的示例仍为 `lafan1_0128_1.onnx`、`folder_path` / `enter_idx`；仓库实际 `config/wbc.json` 为 `dance12_0207_1.onnx`、`motion_path` / `start_idx` / `end_idx`，且 `State_WBC.cpp` 正按后者读取。
- **真实控制链路**：`main.cpp` 设 `dt=0.02`，`ControlFrame` 驱动 `FSM::run`，WBC 经参考数据与本体历史构造观测、ONNX 推理、动作缩放，再走 Unitree DDS LowCmd 输出；`IOSDK` 对 LowState/LowCmd 使用 CRC。
- **复现边界**：只完成官方文档通读与上述核心源码静态核对；未编译、未跑 MuJoCo、未进行 G1 真机测试。无法从部署仓独立核实内置模型的训练配置。
