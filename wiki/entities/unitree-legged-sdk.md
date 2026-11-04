---
type: entity
tags: [repo, unitree, unitreerobotics, sdk, legacy, quadruped, ros1]
status: complete
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unitree-sdk2.md
  - ./unitree-ros.md
  - ./unitree-guide.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/repos/unitree_legged_sdk.md
  - ../../sources/repos/unitree.md
summary: "unitree_legged_sdk 是旧代腿式 UDP SDK（当前文档主支持 Go1）；新机型应优先 unitree_sdk2。保留本页是为了维护 ROS1 真机桥与历史复现，而不是推荐新项目默认依赖。"
---

# unitree_legged_sdk

**unitree_legged_sdk** 主要用于 PC 与运控板之间的通信（亦可在其它机器上经 UDP 使用）。现行 tag（如 v3.8.x）文档写明主支持 **Go1**；Laikago/B1/Aliengo/A1 等需回退历史 release（如 v3.3.1）。

## 一句话定义

SDK1 时代的 UDP 真机通信库——理解 ROS1 `unitree_ros_to_real` 链路时必需，新项目应改走 SDK2。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 本仓为旧代 |
| SDK2 | Unitree SDK version 2 | 新机型默认 |
| UDP | User Datagram Protocol | 本仓通信方式 |
| ROS | Robot Operating System | 常与 ros_to_real 联用 |
| API | Application Programming Interface | C++/可选 Python 封装 |
| Go1 | Unitree Go1 | 当前文档主支持机型 |

## 为什么重要

- 大量 2020–2023 论文与教程仍引用本仓；复现时不能误装 SDK2。
- 与 [`unitree_ros`](./unitree-ros.md) / `unitree_ros_to_real` 组成 ROS1 真机故事线。
- 作为对照，帮助理解为何新栈改 DDS。

## 核心原理

依赖 Boost、CMake、g++；可选 `-DPYTHON_BUILD=TRUE` 编 Python 封装。运行 C++ 示例常需 `sudo` 以锁定内存。固件与 `Legged_sport` 版本需满足 README 下限。

## 工程实践

```bash
mkdir build && cd build && cmake .. && make
# 或
cmake -DPYTHON_BUILD=TRUE ..
```

缺 `msgpack.hpp` 时安装 `libmsgpack*`；Python 在 arm 上需改 `sys.path` 到 `arm64` 库目录。

## 局限与风险

- **机型支持窄**；错用版本会连不上。
- 与 SDK2 **协议不兼容**，禁止在同一控制器里混用。
- 安全：低层示例同样需要吊装与急停流程。

## 源码运行时序图

官方实现 [unitreerobotics/unitree_legged_sdk](https://github.com/unitreerobotics/unitree_legged_sdk)（短 SHA `4539a6c`）：C++ 头文件 + 预编译静态库，示例 `example/example_position.cpp` 通过 UDP 与运控板交换 `LowCmd` / `LowState`，并用 `LoopFunc` 起三个周期线程（接收、控制、发送）。收发与安全限幅的底层实现位于预编译库 `lib/cpp/<arch>/libunitree_legged_sdk.a`，本仓无该部分源码，图中只画头文件暴露的调用关系。

入口：`sudo ./example_position`（由 CMake 目标 `example_position` 编译，C++ 示例需 sudo 以锁内存；运行前按提示确认机器人已挂起并切到低层模式）

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant MAIN as example_position.cpp<br/>main()
    participant CU as Custom<br/>控制类（同文件）
    participant SF as Safety<br/>include/safety.h（实现在预编译库）
    participant UDP as UDP<br/>include/udp.h（实现在预编译库）
    participant LP as LoopFunc<br/>include/loop.h（线程）
    participant RB as 运控板<br/>192.168.123.10:8007
    U->>MAIN: sudo ./example_position
    MAIN->>MAIN: std::cin.ignore()（等待回车确认）
    MAIN->>CU: Custom custom(LOWLEVEL)
    CU->>SF: Safety(LeggedType::Go1)
    CU->>UDP: UDP(LOWLEVEL, 8090, 192.168.123.10, 8007)
    CU->>UDP: InitCmdData(cmd)
    MAIN->>LP: LoopFunc udp_send / udp_recv（绑定 CPU 3）
    MAIN->>LP: LoopFunc control_loop（dt = 0.002）
    MAIN->>LP: start() 三个线程
    par udp_recv 线程（周期 dt）
        LP->>CU: UDPRecv()
        CU->>UDP: Recv()
        UDP->>RB: 接收 LowState 报文
    and control_loop 线程（周期 dt）
        LP->>CU: RobotControl()
        CU->>UDP: GetRecv(state)
        CU->>CU: 计算 qDes / Kp / Kd<br/>（前 10 拍记录初值，10–400 拍线性插值，之后正弦）
        CU->>SF: PositionLimit(cmd)
        CU->>SF: PowerProtect(cmd, state, 1)
        alt 返回值 res1 小于 0
            CU->>MAIN: exit(-1)
        end
        CU->>UDP: SetSend(cmd)
    and udp_send 线程（周期 dt）
        LP->>CU: UDPSend()
        CU->>UDP: Send()
        UDP->>RB: 发送 LowCmd 报文
    end
    MAIN->>MAIN: while(1) sleep(10)（主线程仅保活）
```

图依据：unitree_legged_sdk@4539a6c，读过 `README.md`、`CMakeLists.txt`（`example_position` 目标与库链接）、`example/example_position.cpp`、`include/unitree_legged_sdk/loop.h`、`include/unitree_legged_sdk/udp.h`（grep 声明）、`include/unitree_legged_sdk/safety.h`（grep 声明）、`include/unitree_legged_sdk/comm.h`（grep 片段）、`include/unitree_legged_sdk/unitree_legged_sdk.h`。`UDP` 收发与 `Safety` 限幅的内部实现未读（预编译库，本仓无源码）。

## 关联页面

- [unitree_sdk2](./unitree-sdk2.md)
- [unitree_ros](./unitree-ros.md)
- [unitree_guide](./unitree-guide.md)
- [Unitree](./unitree.md)

## 参考来源

- [sources/repos/unitree_legged_sdk.md](../../sources/repos/unitree_legged_sdk.md)
- 上游：<https://github.com/unitreerobotics/unitree_legged_sdk>
- 源码核对（2026-11-02）：unitreerobotics/unitree_legged_sdk@4539a6c，读过 README.md、CMakeLists.txt、example/example_position.cpp、include/unitree_legged_sdk/loop.h、udp.h（grep）、safety.h（grep）、comm.h（grep）、unitree_legged_sdk.h；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- 开发者文档中 SDK 代际说明：<https://support.unitree.com/home/zh/developer>

