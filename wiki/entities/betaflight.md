---

type: entity
tags: [repo, firmware, fpv, multirotor, uav, open-source, flight-controller, linux-foundation]
status: complete
updated: 2026-09-05
related:
  - ../overview/multirotor-simulation-planning-control-stack.md
  - ./px4-autopilot.md
  - ./gym-pybullet-drones.md
  - ./crazyflie-firmware.md
  - ./wtfos.md
  - ./project-quiver.md
sources:
  - ../../sources/repos/betaflight.md
  - ../../sources/sites/betaflight-com.md
summary: "Betaflight 是 FPV 竞速/自由式多旋翼的开源飞控固件：低延迟姿态环、DShot 电调、Blackbox 与内置 OSD；经 Betaflight App 配置，走 MSP 生态而非 MAVLink/Offboard 自主栈。"
---

# Betaflight

**Betaflight**（[betaflight/betaflight](https://github.com/betaflight/betaflight)）是面向 **FPV 手飞、竞速与自由式** 的 **开源飞控固件**，优化摇杆跟踪与扰动响应；与 [PX4](./px4-autopilot.md) 的 **自主导航 / MAVLink / SITL** 属于多旋翼生态中的 **不同赛道**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FPV | First Person View | 第一人称视角图传飞行 |
| OSD | On-Screen Display | 叠加在图传画面上的飞行数据显示 |
| MSP | Multiwii Serial Protocol | Betaflight 与地面站/外设通信的串口协议族 |
| DShot | Digital Shot | 数字电调协议，低延迟、无需油门校准 |
| IMU | Inertial Measurement Unit | 惯性测量单元，提供加速度与角速度 |
| PID | Proportional–Integral–Derivative | 姿态/速率闭环控制律 |

## 为什么重要

- **FPV 事实标准之一**：全球竞速与自由式社区广泛采用；硬件 Partner 计划覆盖众多 STM32 飞控板。
- **性能导向固件设计**：强调内环延迟、滤波与 rate 曲线，而非任务规划或 Offboard API。
- **可观测性**：**Blackbox** 飞行日志与内置 **OSD** 便于调参与事故复盘；与 [gym-pybullet-drones](./gym-pybullet-drones.md) 中可选的 Betaflight 风格参数可对照理解真机 PID 行为。
- 在多旋翼栈中补齐 **「手飞 FPV 飞控层」**，避免与 PX4 混为同一工具链（见 [多旋翼栈总览](../overview/multirotor-simulation-planning-control-stack.md)）。

## 核心结构/机制

| 组件 | 说明 |
|------|------|
| **姿态 / 速率环** | 多模式（Acro/Angle 等）；飞行中可调 PID、rate profile 与滤波器滑块 |
| **电机输出** | DShot / Multishot / Oneshot 等数字或模拟 PWM 协议 |
| **接收机链路** | SBus、CRSF、PPM 等；内置 failsafe 与解锁安全检查 |
| **MSP** | 与 [Betaflight App](https://app.betaflight.com) 及外设通信；**非 MAVLink** |
| **Blackbox** | 记录陀螺仪、设定点、电机输出等到板载 Flash 或 SD |
| **OSD** | 电量、速度、高度、模式等叠加进模拟图传，无需独立 OSD 芯片固件；经 **MSP** 亦可由 [wtfOS](./wtfos.md) 社区包（如 msp-osd）送入 **DJI 数字图传** |
| **目标硬件** | STM32 F4/G4/F7/H7 等；由社区维护 `target` 配置 |

**版本节奏**（2025 起）：`YYYY.M.PATCH` 语义化版本，每年约 **6 月、12 月** 稳定发布；文档与 App 与固件版本对齐。

```mermaid
flowchart LR
  PILOT["FPV 手飞\n摇杆 / 图传"]
  RX["接收机\nSBus / CRSF"]
  BF["Betaflight\n姿态环 + 混控"]
  ESC["电调\nDShot"]
  MOT["电机"]
  APP["Betaflight App\nMSP 配置"]
  BB["Blackbox\n日志"]

  PILOT --> RX --> BF --> ESC --> MOT
  APP -.->|MSP| BF
  BF --> BB
```

## 常见误区或局限

- **误区：Betaflight = PX4 的轻量版** — Betaflight 服务 **手飞 FPV** 与特技性能；**不提供** MAVLink Offboard、ROS 任务栈或工业级自主导航工作流。
- **误区：仿真里调 Betaflight 参数即可上 PX4 真机** — [gym-pybullet-drones](./gym-pybullet-drones.md) 的「Betaflight 风格」仅为 **近似参数**；与 PX4 SITL 观测/动作空间无关。
- **局限：研究向群体规划** — 多机避障、EGO-Planner 等输出需 **PX4 + MAVSDK** 等伴机栈，而非 Betaflight 原生对接。
- **局限：硬件支持** — 团队不生产飞控板；硬件故障应联系厂商（官方 README 明确说明）。

## 源码运行时序图

入口：固件 `main()`（`src/main/main.c`），由飞控 MCU 上电后执行 `systemInit()`、`initPhase1/2/3()` 完成初始化，再进入 `run()` 的 `scheduler()` 死循环。下图只画源码中飞控主循环的调用顺序（陀螺仪采样、滤波、PID、混控输出、接收机），不含 CLI、MSP 与 OSD 等支路。

```mermaid
sequenceDiagram
    autonumber
    participant MN as main main.c
    participant IN as initPhase1 2 3 init.c
    participant SC as scheduler scheduler.c
    participant GY as taskGyroSample core.c
    participant FT as taskFiltering core.c
    participant PL as taskMainPidLoop core.c
    participant RC as processRcCommand rc.c
    participant PD as pidController pid.c
    participant MX as mixTable writeMotors mixer.c
    participant RX as taskUpdateRxMain tasks.c

    MN->>IN: 依次执行 initPhase1 initPhase2 initPhase3
    IN->>IN: 初始化 mixer motorDevInit 传感器 pidInit imuInit rxInit 等
    MN->>SC: run 进入 while true 循环调用 scheduler
    loop 每次调度循环
        SC->>GY: 实时任务 TASK_GYRO 调用 taskGyroSample
        GY->>GY: gyroUpdate 读取各陀螺仪并求平均
        SC->>FT: TASK_FILTER 调用 taskFiltering
        FT->>FT: gyroFiltering 陀螺仪滤波 可含动态陷波
        SC->>PL: TASK_PID 调用 taskMainPidLoop
        PL->>RC: subTaskRcCommand 把摇杆值换算为设定点
        PL->>PD: subTaskPidController 计算各轴 PID 输出
        PL->>MX: subTaskMotorUpdate 调用 mixTable 混控
        MX->>MX: writeMotors 调用 motorWriteAll 输出电机指令
        PL->>PL: subTaskPidSubprocesses 若启用则 blackboxUpdate
        SC->>RX: TASK_RX 周期或事件触发 taskUpdateRxMain
        RX->>RX: processRx 检查接收机帧与 failsafe 再切换 RX 状态
    end
```

图依据：betaflight/betaflight@3e6a3f8，实际读过 `src/main/main.c`、`src/main/fc/init.c`（initPhase1/2/3 调用段）、`src/main/scheduler/scheduler.c`（scheduler 开头与任务分发）、`src/main/fc/tasks.c`（任务表与 taskUpdateRxMain）、`src/main/fc/core.c`（taskGyroSample、taskFiltering、taskMainPidLoop 及子任务）、`src/main/flight/pid.c`（pidController 签名）、`src/main/flight/mixer.c`（writeMotors、mixTable 签名）、`src/main/fc/rc.c`（processRcCommand 签名）、`src/main/sensors/gyro.c`（gyroUpdate、gyroFiltering）。`motorWriteAll`、`blackboxUpdate` 的内部实现未读，仅按调用点画出。

## 参考来源

- [sources/repos/betaflight.md](../../sources/repos/betaflight.md)
- [sources/sites/betaflight-com.md](../../sources/sites/betaflight-com.md)
- [betaflight/betaflight](https://github.com/betaflight/betaflight)
- [Betaflight 官方站](https://betaflight.com/)
- 源码核对（2026-11-02）：betaflight/betaflight@3e6a3f8，读过 src/main/main.c、src/main/fc/init.c、src/main/scheduler/scheduler.c、src/main/fc/tasks.c、src/main/fc/core.c、src/main/flight/pid.c、src/main/flight/mixer.c、src/main/fc/rc.c、src/main/sensors/gyro.c；时序图依据以上文件，未运行代码。

## 关联页面

- [多旋翼仿真—规划—飞控栈总览](../overview/multirotor-simulation-planning-control-stack.md)
- [PX4 Autopilot](./px4-autopilot.md) — 自主飞控 / MAVLink 对照
- [gym-pybullet-drones](./gym-pybullet-drones.md) — RL 仿真与 Betaflight 风格参数
- [Crazyflie Firmware](./crazyflie-firmware.md) — 微四轴嵌入式栈对照
- [wtfOS](./wtfos.md) — DJI 数字图传端 MSP OSD 与社区包
- [Project Quiver](./project-quiver.md) — 作业级 ArduPilot 机架对照（非 FPV 手飞）

## 推荐继续阅读

- [Betaflight 文档 Wiki](https://betaflight.com/docs/wiki) — 安装、调参、Blackbox
- [Betaflight App](https://app.betaflight.com) — 固件刷写与配置
- [Betaflight 发布说明](https://www.betaflight.com/docs/category/release-notes) — 版本变更
