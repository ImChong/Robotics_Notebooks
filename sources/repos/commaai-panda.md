# panda：车辆 CAN/CAN FD 接口与安全固件 / comma.ai panda — 官方源码归档

- **资料类型：** 开源代码仓库，官方一手资料
- **项目：** https://github.com/commaai/panda
- **仓库分支：** master
- **核对提交：** [4b995ddbebbaaa8ce9fe301ad5a1ae2c140d04ba](https://github.com/commaai/panda/commit/4b995ddbebbaaa8ce9fe301ad5a1ae2c140d04ba)
- **核对日期：** 2026-11-22
- **代码：** https://github.com/commaai/panda（已开源；未本地运行硬件）
- **核对范围：** README.md、board/main.c、board/drivers/fdcan.h、python/base.py；结合 opendbc/opendbc/safety/safety.h
- **通读：** 已完成
- **对应节点：** [wiki/entities/commaai-panda.md](../../wiki/entities/commaai-panda.md)

## 来源所支持的结论

panda 是 comma.ai 的车辆 CAN/CAN FD 硬件与固件组件，面向车载通信接入，并在 MCU 端执行车型相关的安全策略。

官方 README 指出硬件采用 STM32H725，包含 board 固件、python 主机接口、tests 测试。board/main.c 的 set_safety_mode 通过 set_safety_hooks 选择来自 opendbc 的安全模式；失败时回退 SAFETY_SILENT；永久故障强制静默。主机通过 Panda 的 CAN API 收发帧，固件可对拟发送的控制报文实施 safety hooks 和故障处理。

## 实际源码与运行入口

- **核对路径：** board/main.c、board/drivers/fdcan.h、python/base.py、opendbc/safety/safety.h
- **官方快速开始／复现范围：** README 提供 git clone、./setup.sh、./test.sh，Python 端 Panda().can_recv() 与 can_send() 示例；真实汽车使用前需选择匹配车型的安全配置，验证 CAN 总线映射、心跳、故障恢复和 HIL 回归。开放源码不等于目标车辆已经通过安全认证。

## 开源范围与风险边界

已通读官方 README 的硬件、安全、测试与 Python 用法，核对启动安全模式与 MCU 主循环代码；尚未刷机或执行 HIL。

车型安全逻辑已在 opendbc 仓库维护，不能只看 panda 单仓就理解完整限制。测试中的 ALLOUTPUT 等模式不得当成道路安全配置，固件与车型配置需一致；未做硬件验证。

## 上下游关系

- [commaai/openpilot](https://github.com/commaai/openpilot)
- [commaai/msgq](https://github.com/commaai/msgq)
- [commaai/opendbc](https://github.com/commaai/opendbc)

> 本文件只保留核对范围及关键事实，不镜像粘贴官方 README 或受版权保护的整段源码；具体实现以锁定提交为准。
