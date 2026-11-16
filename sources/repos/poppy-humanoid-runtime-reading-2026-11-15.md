# Poppy 平台：官方技术资料与运行入口核对

- **类型：** repo / 官方技术文档
- **核对日期：** 2026-11-15
- **核对范围：** `poppy-project/poppy-humanoid/master` 的根 README、软件启动文档、`software/poppy_humanoid/poppy_humanoid.py` 与 `primitives/safe.py` 全文；并读 `poppy-project/pypot/master/pypot/creatures/abstractcreature.py` 的构造与后台服务路径。以 Humanoid 为具体运行示例，不将其配置移植为全家族通用参数。
- **通读：** 已完成
- **复现状态：** 本次为文档与源码阅读，未连接硬件、未运行控制程序，不等于实机验证。

## 主要原始资料

- [Poppy Humanoid](https://github.com/poppy-project/poppy-humanoid)、[启动说明](https://github.com/poppy-project/poppy-humanoid/blob/master/software/doc/start_Poppy_Humanoid.md)
- [Humanoid setup](https://github.com/poppy-project/poppy-humanoid/blob/master/software/poppy_humanoid/poppy_humanoid.py)、[监控与力矩原语](https://github.com/poppy-project/poppy-humanoid/blob/master/software/poppy_humanoid/primitives/safe.py)
- [pypot creature 工厂](https://github.com/poppy-project/pypot/blob/master/pypot/creatures/abstractcreature.py)。Humanoid 树 `d66c8fbd`。

## 方法、实现与限制

硬件 CC BY-SA4.0，软件 GPLv3，CAD 可能由 Git LFS/Release 提供。`AbstractPoppyCreature.__new__` 读配置，按真实硬件、V-REP 或 dummy 创建 robot，再调用具体 creature 的 setup，最后按选项开后台服务。`PoppyHumanoid.setup` 绑定姿态、空闲、互动等原语；真机启动 TemperatureMonitor，默认0.5Hz、阈值55，超温时播放声音并打印，源码没有由此自动关闭所有关节。LimitTorque 被挂载，但 start 调用被注释，不能声称默认已启用。旧启动说明的系统镜像、登录与 V-REP 路径应按实际设备和依赖复核，不适合作为当前跨平台兼容保证。

## 对 wiki 的映射

- [Poppy 平台](../../wiki/entities/poppy-project-robots.md)
