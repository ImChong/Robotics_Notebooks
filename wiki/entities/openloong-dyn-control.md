---
type: entity
tags: [repo, humanoid, mpc, wbc, mujoco, openloong]
status: complete
updated: 2026-05-27
related:
  - ./openloong.md
  - ../concepts/mpc-wbc-integration.md
  - ../overview/navigation-slam-autonomy-stack.md
sources:
  - ../../sources/repos/openloong_dyn_control.md
  - ../../sources/repos/openloong-dyn-control.md
summary: "OpenLoong-Dyn-Control 是青龙人形 MuJoCo 动力学控制包：MPC+WBC 行走/跳跃/盲踩，与 OpenLoong Framework 并行。"
---

# OpenLoong-Dyn-Control

**OpenLoong-Dyn-Control** 提供青龙人形在 **MuJoCo** 上的 **MPC + 全身控制** 研究与仿真 demo。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| SLAM | Simultaneous Localization and Mapping | 同步定位与建图 |
| EtherCAT | Ethernet for Control Automation Technology | 高实时性工业以太网总线 |

## 为什么重要

- **与导航栈区分**：本仓属 **腿式人形运控**，非 Nav2/SLAM。
- **与 Framework 分工**：Framework 偏实机 C++ 全链；Dyn-Control 偏算法复现与论文级 demo。

## 核心结构/机制

| 能力 | 说明 |
|------|------|
| **MPC** | 质心/足端规划 |
| **WBC** | 关节力矩分配 |
| **MuJoCo** | 仿真验证与盲踩障碍 |

## 常见误区或局限

- **勿与 Autoware/Nav2 混栈** — 输出为关节/力矩级控制，非 `cmd_vel`。
- **硬件**：实机需配合 [OpenLoong](./openloong.md) Framework 与 EtherCAT 驱动。

## 源码运行时序图

入口：`./walk_mpc_wbc`（在 `build/` 目录运行；MuJoCo 仿真，`mj_loadXML("../models/scene.xml")`；其余示例 `walk_wbc`、`jump_mpc` 未读，未画）

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户终端
    participant D as demo/walk_mpc_wbc.cpp main()
    participant MJ as MuJoCo（MJ_Interface）
    participant K as Pin_KinDyn（AzureLoong.urdf）
    participant JS as JoyStickInterpreter
    participant GS as GaitScheduler
    participant FP as FootPlacement
    participant MPC as MPC（dt 0.005 s）
    participant WBC as WBC_priority
    participant PVT as PVT_Ctr
    participant BUS as DataBus RobotState
    U->>D: ./walk_mpc_wbc
    D->>MJ: mj_loadXML + mj_makeData，构造 UIctr 与 MJ_Interface
    D->>K: 加载 AzureLoong.urdf，computeInK_Leg / computeInK_Hand 求初始关节
    D->>WBC: setQini(初始关节)；构造 MPC、GaitScheduler、PVT_Ctr、FootPlacement、JoyStickInterpreter
    loop 渲染窗口未关闭且仿真时间小于 30 s（每帧推进约 1/60 s）
        D->>MJ: mj_step，updateSensorValues，dataBusWrite
        MJ->>BUS: 传感器数据
        D->>K: dataBusRead，computeJ_dJ，computeDyn，dataBusWrite
        K->>BUS: 动力学量
        alt 仿真时间大于 5 s
            D->>JS: setVxDesLPara(1.2, 2.0)，motionState 置为 Walk
        else 之前
            D->>JS: setIniPos（记录当前位姿）
        end
        D->>JS: step，dataBusWrite（x、y、yaw 与速度期望）
        alt 仿真时间不小于 3 s
            D->>GS: step（步态调度）
            D->>FP: getSwingPos（摆腿落脚点）
        end
        alt 每 5 个仿真步（200 Hz）
            D->>MPC: dataBusRead，cal()，dataBusWrite
        end
        D->>WBC: dataBusRead，computeDdq(K)，computeTau，dataBusWrite
        alt 仿真时间不超过 3 s
            D->>BUS: motors_pos_des 取初始站立关节角
        else 之后
            D->>MPC: enable，set_weight
            D->>K: integrateDIY(q, wbc_delta_q_final) 得到期望关节角
            D->>BUS: motors_pos_des / vel_des / tor_des
        end
        D->>PVT: dataBusRead，calMotorsPVT 或对踝、膝 setJointPD，dataBusWrite
        D->>MJ: setMotorsTorque(motors_tor_out)
        D->>D: DataLogger 记录一行
    end
```

图依据：loongOpen/OpenLoong-Dyn-Control@4dd7a7e，读过 `README.md`（构建与运行命令）、`demo/walk_mpc_wbc.cpp`（main 全文）。算法类（MPC、WBC、GaitScheduler 等）的内部实现未读，图只标出 main 中可见的调用。

## 参考来源

- [OpenLoong-Dyn-Control 源码归档](../../sources/repos/openloong-dyn-control.md)（<https://github.com/loongOpen/OpenLoong-Dyn-Control>）

- [sources/repos/openloong_dyn_control.md](../../sources/repos/openloong_dyn_control.md)
- [loongOpen/OpenLoong-Dyn-Control](https://github.com/loongOpen/OpenLoong-Dyn-Control)
- 源码核对（2026-11-02）：loongOpen/OpenLoong-Dyn-Control@4dd7a7e，读过 README.md、demo/walk_mpc_wbc.cpp；时序图依据以上文件，未运行代码。

## 关联页面

- [OpenLoong](./openloong.md)
- [MPC 与 WBC 集成](../concepts/mpc-wbc-integration.md)
- [导航·SLAM·自动驾驶栈总览](../overview/navigation-slam-autonomy-stack.md)

## 推荐继续阅读

- https://www.openloong.org.cn/pages/api/html/index.html
