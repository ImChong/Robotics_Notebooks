# LCM 官方文档（lcm-proj.github.io）

> 来源归档

- **标题：** LCM Documentation
- **类型：** site（官方文档 / 项目页）
- **来源：** LCM Project（起源：MIT DARPA Urban Challenge 团队，2006）
- **链接：** https://lcm-proj.github.io/lcm/
- **代码：** https://github.com/lcm-proj/lcm（已开源，见 [repos/lcm.md](../repos/lcm.md)）
- **入库日期：** 2026-07-28
- **一句话说明：** Lightweight Communications and Marshalling 的权威文档入口：特性、安装、类型语言、UDP 组播协议、日志格式与多语言教程。

## 为什么值得保留

- 本仓库 [lcm-basics](../../wiki/concepts/lcm-basics.md) / [ros2-vs-lcm](../../wiki/comparisons/ros2-vs-lcm.md) 此前主要依赖二手叙述；本页是 **一手官方定义**。
- 明确写出设计边界：**无中心 hub / 无 daemon / 少依赖 / UDP Multicast 广播**——与 ROS 2/DDS 选型对照的关键对照点。
- 提供 Type Spec、UDP Multicast Protocol、Log File format 等协议级链接，便于工程落地与形式化（见 [udp-multicast-dynamics](../../wiki/formalizations/udp-multicast-dynamics.md)）。

## 开源核查（2026-07-28）

| 项 | 状态 |
|----|------|
| 项目页 → GitHub | ✅ 文档 Quick Links / 侧栏指向 [lcm-proj/lcm](https://github.com/lcm-proj/lcm) |
| 代码开放度 | **已开源**（LGPL-2.1；发行版见 GitHub Releases，例如 v1.5.2） |
| 预构建包 | Ubuntu `liblcm-dev` / Homebrew `lcm` / pip `lcm` / AUR / NixOS（见 Installing LCM） |

## 官方自述核心特性（摘录）

1. Low-latency inter-process communication  
2. Efficient broadcast via **UDP Multicast**  
3. Type-safe message marshalling（多语言代码生成）  
4. User-friendly logging and playback  
5. **No centralized database/hub** — peers communicate directly  
6. **No daemons**  
7. Few dependencies  

## 平台与语言（文档站）

| 维度 | 内容 |
|------|------|
| 平台 | GNU/Linux、OS X、Windows、任意 POSIX-1.2001 |
| 语言（文档列出） | C / C++ / C# / Java / Lua / MATLAB / Python（≥3.7） |
| 社区 fork（非官方） | Vala（vooon/lcm-vala）、Rust（adeschamps/lcm） |

> 仓库 README 另将 **Go、C#/.NET** 标为 **unmaintained**（PR 仍欢迎）；以仓库 README 为准做选型。

## 关键文档入口

| 资源 | URL |
|------|-----|
| Installing LCM | https://lcm-proj.github.io/lcm/content/install-instructions.html |
| Build from source | https://lcm-proj.github.io/lcm/content/build-instructions.html |
| Type Specification Language | https://lcm-proj.github.io/lcm/content/lcm-type-specification-language.html |
| UDP Multicast Protocol | https://lcm-proj.github.io/lcm/content/lcm-udp-multicast-protocol-description.html |
| Log File format | https://lcm-proj.github.io/lcm/content/lcm-log-file-format.html |
| UDP Multicast Setup | https://lcm-proj.github.io/lcm/content/udp-multicast-setup.html |
| IROS 2010 Overview PDF | 文档站 Publications（Huang et al.） |
| MIT-CSAIL-TR-2009-041 | 文档站 Technical Report |

## 历史用户（文档站列举，节选）

MIT、CMU、ETH Zurich、Georgia Tech、Google、Ford、Volvo、WHOI、BAE Systems 等——说明 LCM 同时服务研究与量产自动驾驶/机器人。

## 对 wiki 的映射

- 主概念：[lcm-basics](../../wiki/concepts/lcm-basics.md)
- 选型：[ros2-vs-lcm](../../wiki/comparisons/ros2-vs-lcm.md)
- 形式化：[udp-multicast-dynamics](../../wiki/formalizations/udp-multicast-dynamics.md)
- 运控实践：[real-time-control-middleware-guide](../../wiki/queries/real-time-control-middleware-guide.md)
- 代码仓归档：[repos/lcm.md](../repos/lcm.md)
