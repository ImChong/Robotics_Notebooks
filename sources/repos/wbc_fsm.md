# wbc_fsm

> 来源归档（ingest）

- **仓库：** https://github.com/ccrpRepo/wbc_fsm
- **类型：** repo（C++ 部署框架）
- **组织：** ccrpRepo / ZSTU Robotics
- **主语言：** C++ (98.8%)
- **Stars：** 138 | **Forks：** 23
- **入库日期：** 2026-05-01
- **沉淀到 wiki：** 是 → [`wiki/entities/wbc-fsm.md`](../../wiki/entities/wbc-fsm.md)

## 一句话说明

针对 **Unitree G1 人形机器人** 的 C++ 全身控制部署框架，以有限状态机（FSM）管理 Passive / Loco / WBC 三种运行模式，内嵌基于 LAFAN1 动捕数据训练的 ONNX 策略，支持仿真（Unitree Mujoco）与真机（G1 PC2 / aarch64）部署。

## 为什么值得保留

1. **WBC+FSM 部署范式的具体实现**：演示了如何在真实人形机器人上用状态机组织多种控制模式，是 WBC 从算法到系统集成的典型参考。
2. **纯 C++ + Unitree SDK2 + ONNX Runtime**：无 ROS 依赖，轻量易移植，是 Sim2Real ONNX 部署模式的完整示例。
3. **来自 ccrpRepo**：与 AMP_mjlab 同一组织出品，可与 AMP_mjlab 的训练侧形成"训练（Python/mjlab）→部署（C++/FSM）"闭环参考。
4. **LAFAN1 动捕参考驱动**：展示了将 MoCap 数据提炼为 ONNX 策略再上机的完整链路。

## 核心技术摘录

### FSM 状态机结构

| 状态 | 描述 | 进入条件 |
|------|------|---------|
| Passive | 阻尼保护，默认启动状态 | 上电 |
| Loco | 行走 / 移动模式 | START → R2+A |
| WBC | 全身运动跟踪（RL 策略） | R1+Up |

推荐操作顺序：**阻尼 → 站立 → 行走 → WBC**

### 运动模型

- **策略**：`model/wbc/lafan1_0128_1.onnx`（预训练，ONNX Runtime 1.22.0）
- **运动参考**：`motion_data/lafan1/dance12/`（LAFAN1 MoCap → G1 重定向）
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

### 手柄操作

| 按键 | 动作 |
|------|------|
| START | 位置控制（离开阻尼模式） |
| R2 + A | 切换到行走模式 |
| R1 + Up | 切换到 WBC 模式 |
| R2 | 暂停执行 |
| R1 | 恢复 WBC |
| L2 | 当前帧定帧 |

## 与其他 ccrpRepo 项目的关系

```
AMP_mjlab (Python / mjlab)    →    训练出 ONNX 策略
                                         ↓
              wbc_fsm (C++ / Unitree SDK2)  →  部署上机（G1 PC2）
```

wbc_fsm 是 AMP_mjlab 系列的**部署侧对应物**：AMP_mjlab 负责策略训练（Python/IsaacLab API），wbc_fsm 负责策略执行（C++/真机）。

## 当前提炼状态

- [x] 仓库定位与 FSM 状态机结构
- [x] ONNX 部署链路（训练 → ONNX → C++ 推理）
- [x] 与 AMP_mjlab / WBC 体系的关联关系
- [ ] 后续可深挖：config JSON 参数解读、FSM 转移条件形式化、与 unitree_rl_mjlab 部署模式对比
