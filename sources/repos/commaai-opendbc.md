# opendbc：车辆接口与安全规则 / opendbc Car Interface — 官方源码归档

- **资料类型：** 开源代码仓库，官方一手资料
- **项目：** https://github.com/commaai/opendbc
- **仓库分支：** master
- **核对提交：** [34b55aec160a8fa0c5733a9b5279be40b5e1d775](https://github.com/commaai/opendbc/commit/34b55aec160a8fa0c5733a9b5279be40b5e1d775)
- **核对日期：** 2026-11-22
- **代码：** https://github.com/commaai/opendbc（已开源；未本地运行硬件）
- **核对范围：** README.md、examples/joystick.py、opendbc/car/panda_runner.py、opendbc/car/honda/carcontroller.py、opendbc/safety/safety.h
- **通读：** 已完成
- **对应节点：** [wiki/entities/commaai-opendbc.md](../../wiki/entities/commaai-opendbc.md)

## 来源所支持的结论

opendbc 是 comma.ai 的车辆控制接口工程，包含 DBC 信号定义、CAN 报文编解码、车型适配的 Python API 与在 panda 固件使用的安全规则。

opendbc/dbc 保存车型 DBC，opendbc/can 解析/构造 CAN，opendbc/car 中每品牌的 carstate.py、carcontroller.py、interface.py 负责车辆状态和控制映射；opendbc/safety 提供车型安全模型。README 的移植指南要求录制 CAN 轨迹、分析信号、实现车型目录并按清单验证。PandaRunner 选择车型与安全模式，read 调用 CI.update，write 调用 CI.apply 生成 CAN，并经 panda 发送。

## 实际源码与运行入口

- **核对路径：** examples/joystick.py、opendbc/car/panda_runner.py、opendbc/car/honda/carcontroller.py、opendbc/safety/safety.h
- **官方快速开始／复现范围：** 官方 quickstart 为 ./test.sh；示例 examples/joystick.py 通过 PandaRunner 实现键盘/摇杆输入的 100 Hz 指令示范（仅在有合适车辆与硬件的测试环境运行）。移植时先做 fingerprint / DBC 解析，再实现 carstate 与 carcontroller，使用官方 CAR_PORTING_CHECKLIST 验证，不应直接照搬别的车型控制报文。

## 开源范围与风险边界

已通读官方 README 项目结构、移植流程与安全模式说明，核对 joystick 示例、PandaRunner、Honda 控制器和 safety.h；未在实际车辆上运行。

不同车型能读取和能控制的信号不同；README 明确默认 SAFETY_SILENT，只有选择相应安全模式后才可能发送。所谓统一 API 不代表支持所有车型、也不代表安全规则可以由上位机忽略。

## 上下游关系

- [commaai/openpilot](https://github.com/commaai/openpilot)
- [commaai/msgq](https://github.com/commaai/msgq)
- [commaai/panda](https://github.com/commaai/panda)

> 本文件只保留核对范围及关键事实，不镜像粘贴官方 README 或受版权保护的整段源码；具体实现以锁定提交为准。
