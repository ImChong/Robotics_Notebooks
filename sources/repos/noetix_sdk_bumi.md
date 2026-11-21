# noetix_sdk_bumi

> 来源归档（国内具身开源全景）

- **标题：** noetix_sdk_bumi
- **类型：** repo
- **机构：** 松延动力
- **链接：** https://github.com/Noetix-Robotics/noetix_sdk_bumi
- **分类：** SDK/驱动
- **入库日期：** 2026-09-06
- **核对范围：** 已克隆 `Noetix-Robotics/noetix_sdk_bumi@95b2645`（核对 `README.md`、`CMakeLists.txt`、`include/lowcontroller.h`、`examples/cpp/example_policy.cpp`；未在本地连接 Bumi 实体机器人及 CycloneDDS 总线运行）。
- **通读：** 已完成
- **一句话说明：** 松延动力小布米（Bumi）双足人形机器人的 C++17 / Python 底层与高层控制 SDK，封装预编译 `libnoetix_sdk.a` 与 CycloneDDS 通信，提供 500 Hz 底层力矩控制、2 ms（2000 us）RL 策略部署（WALK/DANCE ONNX 切换）与多媒体交互示例。
- **沉淀到 wiki：** [`wiki/entities/cn-os-noetix-sdk-bumi.md`](../../wiki/entities/cn-os-noetix-sdk-bumi.md)

## 开源状态

- **已开源（示例与头文件开源，核心库为预编译静态库）**：公开仓库（<https://github.com/Noetix-Robotics/noetix_sdk_bumi>）；用户态示例、`include/*.h` 与 pybind11 绑定源码公开，底层通信/固件桥接以 `lib/<arch>/libnoetix_sdk.a` 及 CycloneDDS 预编译二进制交付。

## 一手仓库要点摘录（`Noetix-Robotics/noetix_sdk_bumi@95b2645`）

- **构建目标（`CMakeLists.txt`）**：支持 Ubuntu 22.04（GCC 11.4.0，x86_64/aarch64）；编译 C++ 示例 `example_high`、`example_low`、`example_policy`、`example_media`，并打包 pybind11 模块 `highcontrol_py`、`lowcontrol_py`、`mediacontrol_py`。
- **更新核对（`95b2645`）**：原 `example_policy_walk.cpp` 已重构更名为 `examples/cpp/example_policy.cpp`，新增 WALK 与 DANCE 两套 ONNX 模型动态切换，`kLoopPeriodUs=2000`（2 ms，500 Hz），`kNumJoints=21`，摔倒重力阈值 `kFallGravityZ=-0.3`，主类涵盖 `loadModel()`、`updateStateEstimation()`、`computeObservation()`、`computeActions()`、`handleUserMode()`、`process()`。

## 对 wiki 的映射

- [wiki/entities/cn-os-noetix-sdk-bumi.md](../../wiki/entities/cn-os-noetix-sdk-bumi.md)
