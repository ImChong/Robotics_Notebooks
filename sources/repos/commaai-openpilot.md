# openpilot：模块化驾驶辅助系统 / openpilot Modular Driver Assistance — 官方源码归档

- **资料类型：** 开源代码仓库，官方一手资料
- **项目：** https://github.com/commaai/openpilot
- **仓库分支：** master
- **核对提交：** [4bd73298e9a820c64426b973e81675c702a2e650](https://github.com/commaai/openpilot/commit/4bd73298e9a820c64426b973e81675c702a2e650)
- **核对日期：** 2026-11-22
- **代码：** https://github.com/commaai/openpilot（已开源；未本地运行硬件）
- **核对范围：** README.md、openpilot/cereal/README.md、launch_chffrplus.sh、openpilot/system/manager/manager.py、openpilot/selfdrive/modeld/modeld.py、openpilot/selfdrive/controls/controlsd.py、openpilot/selfdrive/car/card.py
- **通读：** 已完成
- **对应节点：** [wiki/entities/commaai-openpilot.md](../../wiki/entities/commaai-openpilot.md)

## 来源所支持的结论

openpilot 是 comma.ai 面向受支持汽车的开源驾驶辅助系统，以独立进程组合感知、驾驶模型、状态判定、控制及车辆接口。

启动入口 launch_chffrplus.sh 建立 Python 环境、准备子模块后运行 manager.py；manager 根据设备状态启停受管理进程。cereal 使用 msgq 发布订阅传输消息，Cap’n Proto 定义序列化结构。controlsd 订阅 modelV2、车辆状态和控制相关消息，计算转向、加速度等 CarControl；card.py 经 opendbc 的车型接口将车辆 CAN 状态解析并生成可发送的控制报文。panda 的车端安全规则不由 Python 控制器绕过。

## 实际源码与运行入口

- **核对路径：** openpilot/system/manager/manager.py、openpilot/selfdrive/controls/controlsd.py、openpilot/selfdrive/car/card.py
- **官方快速开始／复现范围：** 研究入口：先读 launch_chffrplus.sh → manager.py / process_config.py → cereal README → controlsd.py → card.py；车型支持与安装前提以官方 docs/CARS.md 为准。官方 README 给出设备安装入口 openpilot.comma.ai 及开发环境安装脚本。

## 开源范围与风险边界

已核对官方完整 README、cereal 消息规范、启动脚本与控制主进程的可执行入口；未实车验证。

官方定义是驾驶辅助，不是可放手无人驾驶系统；支持范围依车型、车型接口和硬件线束而变化。官方 README 标明软件用于研究、无保修并要求遵守当地法律；源码架构可读不代表已进行实车安全验证。

## 上下游关系

- [commaai/msgq](https://github.com/commaai/msgq)
- [commaai/panda](https://github.com/commaai/panda)
- [commaai/opendbc](https://github.com/commaai/opendbc)

> 本文件只保留核对范围及关键事实，不镜像粘贴官方 README 或受版权保护的整段源码；具体实现以锁定提交为准。
