# lcm-proj/lcm

> 来源归档

- **标题：** Lightweight Communications and Marshalling (LCM)
- **类型：** repo
- **来源：** LCM Project（`lcm-proj`）；起源 MIT DARPA Urban Challenge（2006）
- **链接：** https://github.com/lcm-proj/lcm
- **文档：** https://lcm-proj.github.io/lcm/（归档：[sites/lcm-proj-github-io.md](../sites/lcm-proj-github-io.md)）
- **Stars：** ~1.2k（2026-07）
- **Forks：** ~427
- **默认分支：** `master`
- **最新发行：** v1.5.2（2025-10-23）
- **许可证：** LGPL-2.1
- **入库日期：** 2026-07-28
- **一句话说明：** 面向高带宽、低延迟实时系统的 pub/sub + 类型安全 marshalling 库；UDP 组播、无 daemon、多语言绑定。
- **沉淀到 wiki：** 是 → [`wiki/concepts/lcm-basics.md`](../../wiki/concepts/lcm-basics.md)

## 开源状态（2026-07-28）

**已开源**：完整库、工具、多语言绑定与发行包；项目页与 README 互指。

## README 定位（摘要）

- 目标：real-time systems where **high-bandwidth and low latency** are critical。
- 模型：publish/subscribe + automatic marshalling/unmarshalling code generation。
- Roadmap（官方）：项目 **active again**；近期以 **stability / maintenance** 为主，长期可演进但要求 **backwards compatibility**。

## 特性清单（与文档站对齐）

| 特性 | 说明 |
|------|------|
| Low-latency IPC | 进程间低延迟通信 |
| UDP Multicast | 高效广播，无中心 hub |
| Type-safe marshalling | `.lcm` 类型 → 各语言生成代码 |
| Logging / playback | `lcm-logger` / logplayer 等工具链 |
| No daemons | 无后台守护进程 |
| Few dependencies | 依赖面小，易嵌入运控机 |

## 支持矩阵（README，2026-07）

| 平台 | 版本提示 |
|------|----------|
| Ubuntu | 22.04、24.04 |
| Fedora | 42 |
| macOS | 14、15 |
| Windows | 2019、2022 |

| 语言 | 状态 |
|------|------|
| C / C++ / Java / Lua / MATLAB / Python（≥3.7） | 维护中 |
| Go / C#/.NET | **Unmaintained**（仍接受 PR） |

## 安装入口（官方 Quick Links）

| 方式 | 入口 |
|------|------|
| 文档安装页 | https://lcm-proj.github.io/lcm/content/install-instructions.html |
| 源码构建 | https://lcm-proj.github.io/lcm/content/build-instructions.html |
| Releases | https://github.com/lcm-proj/lcm/releases |
| Ubuntu apt | `sudo apt install liblcm-dev`（Java：`liblcm-java`；Noble：`python3-lcm`） |
| Homebrew | `brew install lcm` |
| pip | `pip3 install lcm`（含 Python 模块与部分 CLI；依赖 GLib 2.0；musl 发行版无 Java GUI） |

## 对 wiki 的映射

- [LCM 基础](../../wiki/concepts/lcm-basics.md)
- [ROS 2 vs LCM](../../wiki/comparisons/ros2-vs-lcm.md)
- [UDP 组播动力学](../../wiki/formalizations/udp-multicast-dynamics.md)
- [实时运控中间件配置指南](../../wiki/queries/real-time-control-middleware-guide.md)
- 项目页：[lcm-proj-github-io.md](../sites/lcm-proj-github-io.md)

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [ros2.md](ros2.md) / [ros2-official-documentation.md](../sites/ros2-official-documentation.md) | 中高层生态 vs 底层高频运控对照 |
| [dimensionalos_dimos.md](dimensionalos_dimos.md) | DimOS 默认 LCMTransport |
| [jackhan / Yobotics E3 LCM 模板](../../wiki/entities/jackhan-yobotics-e3-algorithm-template.md) | 人形外接算法 LCM 对接实例 |
