---
type: concept
tags: [micromouse, maze, embedded, education, competition, path-planning, stm32, differential-drive, ukmars]
status: complete
updated: 2026-07-27
related:
  - ../entities/ukmarsbot.md
  - ../entities/wolfiemouse.md
  - ../methods/a-star.md
  - ../methods/pid-control.md
  - ../entities/kicad.md
  - ../entities/mushr.md
sources:
  - ../../sources/sites/micromouseonline-com.md
  - ../../sources/sites/ukmars-org.md
  - ../../sources/repos/ukmarsbot.md
  - ../../sources/repos/wolfiemouse.md
  - ../../sources/repos/opatiny-micromouse.md
  - ../../sources/repos/lime7git-micromouse.md
  - ../../sources/repos/emstef-micromouse.md
  - ../../sources/repos/ianmhoffman-micromouse.md
  - ../../sources/courses/opatiny_algernon_remote_debug_youtube.md
  - ../../sources/courses/ukmarsbot_getting_started_youtube.md
summary: "Micromouse 是自 1970 年代末延续的迷宫竞速：自主差速鼠在未知 16×16（或半尺寸 32×32）迷宫中建图并尽快抵达中心；栈为传感定位→建图→Flood Fill/A*→运动控制。"
---

# Micromouse

## 一句话定义

**Micromouse** 是要求 **完全自主** 的小型差速（或同类）机器人在 **事先未知的网格迷宫** 中探索建图，并在后续跑次中以 **最短时间** 抵达中心目标区的经典机器人竞赛——现代规则谱系可追溯到 1977 IEEE Spectrum 构想与 1980 前后「找中心」规则定型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Micromouse | Micromouse competition | 自主迷宫竞速机器人赛事总称 |
| Flood Fill | Flood-fill maze search | 以目标为 0 向全图扩散距离场，沿梯度走最短格路径 |
| A\* | A-star | 带启发的图搜索；竞赛鼠常用变体之一 |
| ToF | Time of Flight | 飞行时间测距（墙检/侧墙跟随，如 VL6180X） |
| IR | Infrared | 反射式红外墙传感（经典方案） |
| MCU | Microcontroller Unit | 机载主控（Arduino / STM32 / ESP32 / Pico 等） |
| PCB | Printed Circuit Board | 常兼作底盘结构件 |

## 为什么重要

- **微型「感知—建图—规划—控制」全栈**：在一枚 MCU 上闭环完成 localization、mapping、path planning、motion control，是嵌入式移动机器人的压缩教材。
- **工程可动手入口丰富**：从 [UKMARSBOT](../entities/ukmarsbot.md) 入门板到 [WolfieMouse](../entities/wolfiemouse.md) 竞赛鼠，再到 Webots / Qt 仿真，学习曲线可分段。
- **与现代 AMR 对照**：无 ROS、无 LiDAR 建大图，却共享「搜索跑 vs 竞速跑、最短 ≠ 最快、墙检校正里程计」等思想，可锚定到 [A\*](../methods/a-star.md) / [PID](../methods/pid-control.md)。

## 核心原理

### 规则直觉（经典 / 半尺寸）

| 项目 | 经典 Classic | 半尺寸 Half-size |
|------|--------------|------------------|
| 网格 | 16×16 | 32×32 |
| 格宽 | ~180 mm | ~90 mm |
| 起点 | 角格 | 角格 |
| 目标 | 中心 2×2 | 中心区域 |
| 机身上限 | 约 25×25 cm（无高限，各地细则略异） | 约 12.5×12.5 cm |

细则以赛会文本为准；[Micromouse Online](https://micromouseonline.com/) 与 [UKMARS](https://ukmars.org/) 提供常用英制/欧制叙述。

### 软件四件套

```mermaid
flowchart LR
  SENSE["墙传感 / 编码器 / IMU"] --> LOC["定位 · 格坐标"]
  LOC --> MAP["墙壁地图"]
  MAP --> PLAN["Flood Fill / A* / 变体"]
  PLAN --> MOT["速度环 · 转弯 · 侧墙跟随"]
  MOT --> SENSE
```

1. **搜索跑（search）**：从角格出发探索，把墙壁写入地图；可用 Flood Fill 边走边更新到中心的距离场。
2. **竞速跑（speed run）**：在已知（或部分已知）地图上选 **时间最优** 路径——少转弯、已知直道加速、平滑转弯 / 对角线往往比「格数最短」更快。
3. **运动层**：差速轮速 [PID](../methods/pid-control.md)、侧墙跟随、原地转 vs 平滑弧线转；打滑时用墙缝周期校正位姿。

### 开源与教学资源地图（本次 ingest）

| 资源 | 角色 | 开源要点 |
|------|------|----------|
| [UKMARSBOT](../entities/ukmarsbot.md) | 协会入门多用途底盘 + 教程视频 | MIT 硬件；mazerunner-core 迷宫固件 |
| [WolfieMouse](../entities/wolfiemouse.md) | IEEE Region 1 竞赛级 STM32 + KiCad | 固件/仿真/PCB/工具公开 |
| [Algernon / opatiny](../../sources/repos/opatiny-micromouse.md) | 学士论文：ESP32-S3 + Wi‑Fi 调试 | CERN-OHL-P-2.0 + 固件分仓 |
| [lime7git](../../sources/repos/lime7git-micromouse.md) | 硕士：STM32 + EasyEDA + **Qt 仿真器** | 仓公开；根 LICENSE 待确认 |
| [Webots 项目](../../sources/repos/emstef-micromouse.md) | 仿真四件套 + Flood Fill | GitHub + 项目页 |
| [Ian Hoffman 设计日志](../../sources/sites/ian-hoffman-micromouse-github-io.md) | 规则 / 硬件选型笔记 | 文档公开；成品交付进行中 |
| [Micromouse Online](https://micromouseonline.com/) | 经典技术长文 | 站点公开（非 IEEE 法人官网） |
| [UKMARS](https://ukmars.org/) | 赛事与平台社区 | 与 GitHub `ukmars` 互指 |

## 工程实践

| 阶段 | 建议 |
|------|------|
| **0 仿真** | [Webots Micromouse](https://emstef.github.io/Micromouse/) 或 lime7 Qt 仿真器先跑通 Flood Fill / 地图数组 |
| **1 入门真机** | 跟 [UKMARSBOT 视频](https://www.youtube.com/watch?v=kLJjyr8uiFg) 组装；再刷 [mazerunner-core](https://github.com/ukmars/mazerunner-core) |
| **2 传感标定** | 红外/ToF 暗电流与墙距曲线；编码器极性与轮径；侧墙跟随增益 |
| **3 竞赛向** | 读 [WolfieMouse](../entities/wolfiemouse.md) 与 [Micromouse Online](https://micromouseonline.com/)：平滑转弯、已知格加速、对角线 |
| **4 PCB** | 入门可外购模块；自研底盘用 [KiCad](../entities/kicad.md)（WolfieMouse / Algernon）或 EasyEDA（lime7） |
| **调试** | 蓝牙串口（UKMARSBOT）或 Wi‑Fi Web UI（[Algernon 演示](https://www.youtube.com/watch?v=nz4QlaSIkbY)） |

## 局限与风险

- **不是 ROS/Nav2 导航**：格地图 + 嵌入式实时环，与 [MuSHR](../entities/mushr.md) / 室内 AMR 栈工具不同；概念可迁移，代码勿硬套。
- **最快 ≠ 最短**：只优化格距离会在多转弯迷宫输给「少转高速」策略。
- **规则地区差**：评分（含搜索时间惩罚等）因赛会而异；以当场规则为准。
- **许可注意**：lime7 / 部分课程仓可能缺根 LICENSE；商用或再发布前自行确认。
- **命名消歧**：`micromouseonline.com` 是 Peter Harrison 经典站，**不要**写成 IEEE 官方门户（IEEE 举办/赞助过多届赛事，但本站属社区技术站）。

## 关联页面

- [UKMARSBOT](../entities/ukmarsbot.md) — 推荐起步平台
- [WolfieMouse](../entities/wolfiemouse.md) — 竞赛级全栈参考
- [A* 全局路径规划](../methods/a-star.md) — 与 Flood Fill 同属离散搜索族
- [PID Control](../methods/pid-control.md) — 轮速 / 侧墙跟随底层
- [KiCad](../entities/kicad.md) — 自研 PCB 主路径
- [MuSHR](../entities/mushr.md) — 另一类低成本教育移动平台（ROS 竞速）

## 参考来源

- [sources/sites/micromouseonline-com.md](../../sources/sites/micromouseonline-com.md)
- [sources/sites/ukmars-org.md](../../sources/sites/ukmars-org.md)
- [sources/repos/ukmarsbot.md](../../sources/repos/ukmarsbot.md)
- [sources/repos/wolfiemouse.md](../../sources/repos/wolfiemouse.md)
- [sources/repos/opatiny-micromouse.md](../../sources/repos/opatiny-micromouse.md)
- [sources/repos/lime7git-micromouse.md](../../sources/repos/lime7git-micromouse.md)
- [sources/repos/emstef-micromouse.md](../../sources/repos/emstef-micromouse.md)
- [sources/repos/ianmhoffman-micromouse.md](../../sources/repos/ianmhoffman-micromouse.md)
- [sources/courses/opatiny_algernon_remote_debug_youtube.md](../../sources/courses/opatiny_algernon_remote_debug_youtube.md)
- [sources/courses/ukmarsbot_getting_started_youtube.md](../../sources/courses/ukmarsbot_getting_started_youtube.md)

## 推荐继续阅读

- [Micromouse Online](https://micromouseonline.com/) — 控制 / 传感 / 高速跑长文
- [UKMARS 官网](https://ukmars.org/) — 赛事与入门资源
- [Veritasium: The Fastest Maze-Solving Competition on Earth](https://www.youtube.com/watch?v=ZMQbHMgK2rw) — 协会首页常推的历史与现状纪录片（外链）
