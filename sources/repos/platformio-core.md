# PlatformIO Core

> 来源归档（ingest）

- **标题：** PlatformIO Core — 跨平台嵌入式开发命令行工具
- **类型：** repo（Python CLI、构建系统与包管理器）
- **维护方：** PlatformIO
- **链接：** <https://github.com/platformio/platformio-core>
- **官网：** <https://platformio.org/>
- **官方文档：** <https://docs.platformio.org/en/latest/core/>
- **入库日期：** 2026-10-22
- **GitHub 指标快照：** 9,551 stars、913 forks（GitHub API，2026-10-22；指标会变化）
- **许可：** Apache-2.0（GitHub 仓库元数据）
- **开源状态：** **已开源** — 官方 Core 仓库公开，包含 `platformio/`、`tests/`、示例和 Apache-2.0 LICENSE。
- **一句话说明：** 用统一的 `pio` CLI 和 `platformio.ini` 描述嵌入式项目环境，管理平台/工具链/库，并执行构建、上传、串口监视与单元测试。
- **沉淀到 wiki：** [PlatformIO Core](../../wiki/entities/platformio-core.md)

## 为何值得保留

- 机器人执行器控制板、传感器节点和通信模块常由不同 MCU 与 SDK 构成；PlatformIO 将板卡、框架和依赖声明收进项目配置，便于复现固件构建环境。
- 统一 CLI 可嵌入桌面 IDE、脚本或 CI；其价值是管理嵌入式开发流程，不是代替 MCU 固件、实时操作系统或机器人控制算法。
- 其项目配置、包与库依赖管理可作为机器人底层固件工程化的参考。

## 官方资料摘要

- **Core 职责：** 官方文档列出多平台构建系统、统一包管理器、库管理、库依赖查找器、串口监视器，以及与 IDE 和 CI 的集成。
- **项目配置：** `platformio.ini` 声明项目环境；`pio project init` 可初始化项目结构，按环境执行 `pio run`。
- **测试：** `pio test` 运行 PlatformIO Unit Testing，可在主机本地执行测试，也支持面向嵌入式目标的测试工作流。
- **支持范围：** Core 使用 Python，覆盖 Windows、macOS、Linux、FreeBSD 和 ARM 单板机等平台。具体板卡、框架与调试支持以 PlatformIO Development Platforms 和设备清单为准。

## 对 wiki 的映射

| 主题 | 页面 |
|------|------|
| 工具实体 | [wiki/entities/platformio-core.md](../../wiki/entities/platformio-core.md) |
| 浏览器端 MCU 外设仿真对照 | [wiki/entities/wokwi.md](../../wiki/entities/wokwi.md) |
| 机器人电机固件实践背景 | [wiki/entities/simplefoc.md](../../wiki/entities/simplefoc.md) |

## 参考链接

- [PlatformIO Core 源码与许可证](https://github.com/platformio/platformio-core)
- [PlatformIO Core 官方文档](https://docs.platformio.org/en/latest/core/)
- [PlatformIO Core CLI 安装页](https://platformio.org/install/cli)
- [pio project init](https://docs.platformio.org/en/latest/core/userguide/project/cmd_init.html)
- [pio run](https://docs.platformio.org/en/latest/core/userguide/cmd_run.html)
- [pio test](https://docs.platformio.org/en/latest/core/userguide/cmd_test.html)
