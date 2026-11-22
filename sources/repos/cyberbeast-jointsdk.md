# CyberBeast JointSDK：纯 C99 关节协议栈 / Portable Joint Protocol SDK

> 来源归档（CyberBeast 官方 GitHub）

- **标题：** JointSDK
- **类型：** repo / firmware / joint-control
- **官方仓库：** https://github.com/cyberbeast0829/JointSDK
- **核对分支：** `main`
- **入库日期：** 2026-11-22
- **核对范围：** README 核心设计约束、构建、客户侧例子、后端/端点与安全机制章节；examples/03_control_loop.c 全部执行路径。一手入口：`README.md`、`examples/03_control_loop.c`、`include/joint_sdk.h`；审查重点为本次软件栈定位，不声称覆盖仓库全部源码、历史提交或硬件实测。
- **通读：** 已完成
- **一句话说明：** 将 CYBERBEAST CAN/CAN-FD 帧和运行时端点描述符封装为物理量级 joint API 的跨平台 C99 SDK，控制路径零 malloc。
- **开源状态：** 已公开源码；对应构建与真机兼容性以仓库 README 和具体硬件版本为准。
- **沉淀到 wiki：** 是 → [cyberbeast-jointsdk](../../wiki/entities/cyberbeast-jointsdk.md)

## 一手事实与代码定位

调用者提供 arena 和 HAL send/recv/now_ms，在 configure 阶段握手并下载 JSON 描述符，activate 后每拍 cycle_begin→jsdk_joint_set_mit→cycle_end；CLI 支持 virtual、slcan、socketcan、pcan，默认不开堆模式。

## 官方构建 / 运行入口

- `cmake -S . -B build -DJSDK_WERROR=ON`
- `cmake --build build && (cd build && ctest)`
- `./build/jsdk-cli --if virtual scan`
- `./build/examples/03_control_loop`

## 工程边界

端点 ID 必须由当前设备的 JSON 描述符解析，不能硬编码为 0；1 kHz 示例采用虚拟后端，不等于真机总线实时保证；SLCAN 不适合高频生产循环。

## 参考原文

- [README.md](https://github.com/cyberbeast0829/JointSDK/blob/main/README.md)
- [examples/03_control_loop.c](https://github.com/cyberbeast0829/JointSDK/blob/main/examples/03_control_loop.c)
- [include/joint_sdk.h](https://github.com/cyberbeast0829/JointSDK/blob/main/include/joint_sdk.h)
