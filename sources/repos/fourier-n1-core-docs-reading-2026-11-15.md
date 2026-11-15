# Fourier GRX N1：官方技术资料与运行入口核对

- **类型：** repo / 官方技术文档
- **核对日期：** 2026-11-15
- **核对范围：** `FFTAI/fourier-grx-N1/main` 的 N1 快速开始、运行模式、User/Developer 示例章节、PubSub 协议全文；`FFTAI/Wiki-GRx-Deploy/FourierN1` 的 `user/demo_servo_on.py` 与 `user/demo_walk.py` 全文。未阅读未公开的控制器内部实现。
- **通读：** 已完成
- **复现状态：** 本次为文档与源码阅读，未连接硬件、未运行控制程序，不等于实机验证。

## 主要原始资料

- [N1 快速开始](https://github.com/FFTAI/fourier-grx-N1/blob/main/docs/quickstart/n1.md)、[运行模式](https://github.com/FFTAI/fourier-grx-N1/blob/main/docs/reference/run_type.md)
- [User 示例](https://github.com/FFTAI/fourier-grx-N1/blob/main/docs/examples/user.md)、[Developer 示例](https://github.com/FFTAI/fourier-grx-N1/blob/main/docs/examples/developer.md)、[PubSub 协议](https://github.com/FFTAI/fourier-grx-N1/blob/main/docs/reference/pubsub.md)
- [使能入口](https://github.com/FFTAI/Wiki-GRx-Deploy/blob/FourierN1/user/demo_servo_on.py)、[手柄示例](https://github.com/FFTAI/Wiki-GRx-Deploy/blob/FourierN1/user/demo_walk.py)。文档树 `aa06d0a1`，示例树 `4de14209`。

## 方法、实现与限制

`sdk` 模式对应 User API，`server` 对应 PubSub，Developer 底层示例要求机载运行。4.0.0 以上 Developer 使用 `developer_radian`，旧目录示例以度为单位。User 使能示例导入 `fourier_grx.sdk.user`，创建 Zenoh 会话，向 `fourier-grx/dynalink_interface/task/client` 发布 msgpack 任务命令；手柄示例将虚拟外设轴发往 `grx/client`。这与 PubSub 的 `robot/control` 和 `task/control` 是不同接口，不能只换 topic 而复用消息字典。PubSub 文档要求按关节顺序填 23 项控制模式数组，4.0.0 以上位置/速度单位为 rad/rad/s。User 文档还保留模块消息误发的说明，但本次所读 `demo_walk.py` 已将模块指令发往 `task/client`，复现时应固定文档与示例版本。文档站不是 SDK 完整源码或 BOM 的发布证明。

## 对 wiki 的映射

- [Fourier GRX N1](../../wiki/entities/fourier-grx-n1.md)
