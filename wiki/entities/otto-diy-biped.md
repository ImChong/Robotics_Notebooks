---
type: entity
tags: [biped, education, open-source, arduino, 3d-printing]
status: complete
updated: 2026-11-02
project_id: otto-diy-biped
code: https://github.com/OttoDIY/OttoDIYLib
summary: "Otto DIY：面向教育的低成本 Arduino 兼容开源双足机器人，全 3D 打印，4 舵机基本版，Blockly 图形编程，MIT/GPL 为主。"
related:
  - ./humanoid-robot.md
  - ./open-source-humanoid-hardware.md
  - ../overview/robot-open-source-wechat-githubstore-18-projects.md
  - ../overview/open-source-buildable-humanoids-19-technology-map.md
sources:
  - ../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md
  - ../../sources/blogs/wechat_githubstore_19_buildable_open_source_humanoids_2026-11-02.md
---

# Otto DIY

## 一句话定义

**Otto DIY** 是面向教育的低成本开源双足机器人，官方站点 ottodiy.com，库仓库 [OttoDIY/OttoDIYLib](https://github.com/OttoDIY/OttoDIYLib)，组织入口 [OttoDIY](https://github.com/OttoDIY)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Blockly | Blockly visual programming | 积木式图形化编程 |
| STL | Stereolithography file | 3D 打印模型格式 |
| MIT | MIT License | 宽松开源许可证 |
| GPL | GNU General Public License | 著佐权开源许可证 |

## 为什么重要

- 入门门槛最低的双足平台，作为本批 18 项中「第一台机器人」的起点。
- 在微信文章《想进入机器人领域吗？学习这18个开源项目就够了》中排第 **4** 位。

## 核心要点（据文章）

- Arduino 兼容，结构件全 3D 打印；基本版使用 4 个舵机控制腿部，可扩展传感器和表情模块。
- 提供完整 3D 文件、Arduino 库（OttoDIYLib）、Blockly 图形化编程、教程和纸模；衍生版本 Otto Ninja 支持行走+滚动。
- GitHub 上有多个活跃仓库（OttoDIYLib、OttoDIY 组织、Open-Hardware-Leaders/OttoDIY），许可证以 MIT/GPL 为主。

## 工程实践

- 适合第一次做机器人的人：打印 → Arduino 接线 → Blockly 调动作 → 再换 OttoDIYLib 写代码。
- 以上规格与许可证均出自文章叙述，**未逐项核对官方仓库**；动手前请以官方 README/BOM 为准。

## 机制与架构图

```mermaid
flowchart LR
    Res["开源资源 (3D 打印 STL / 纸模 / OttoDIYLib / Blockly)"] --> Build["Arduino 兼容主控 + 4 舵机腿部基本版"]
    Ext["可选扩展 (传感器 / 表情模块 / Otto Ninja 衍生版)"] --> Build
    Flow["入门路径: 打印 -> 接线 -> Blockly 调动作 -> OttoDIYLib 编程"] --> Build
```

## 源码运行时序图

```mermaid
sequenceDiagram
    autonumber
    participant Sketch as "示例草图 (Otto_avoid.ino / Otto_APP.ino)"
    participant SCmd as "SerialCommand (src/SerialCommand.cpp)"
    participant Otto as "Otto (src/Otto.cpp)"
    participant Osc as "Oscillator[4] (src/Oscillator.cpp)"
    participant HW as "4 路舵机与外设 (EEPROM / 蜂鸣器 / 点阵)"

    Sketch->>Otto: "setup(): Otto.init(YL, YR, RL, RR, true, Buzzer)"
    Otto->>Osc: "attachServos() -> servo[0..3].attach(pin)"
    Otto->>HW: "EEPROM.read(0..3) 读取校准微调 -> SetTrim()"
    Sketch->>Otto: "Otto.home() 归位到 90° 中位"

    loop loop() 主循环
        alt 蓝牙串口控制 (Otto_APP.ino)
            Sketch->>SCmd: "SCmd.readSerial() 解析 M/S/H/K 指令"
            SCmd-->>Sketch: "receiveMovement() 更新 moveId / T / moveSize"
        else 超声波避障 (Otto_avoid.ino)
            Sketch->>HW: "ultrasound() 测距 (<= 15 cm 触发避障)"
        end
        Sketch->>Otto: "Otto.walk(steps, T, dir) / turn() / _moveServos()"
        Otto->>Osc: "_execute() -> oscillateServos() 配置 O/A/T/phase_diff"
        loop 周期 T 内循环采样
            Otto->>Osc: "servo[i].refresh()"
            Osc->>Osc: "next_sample() 到达 _samplingPeriod (默认 30 ms) 后计算 sin 相位与限速"
            Osc->>HW: "_servo.write(_pos + _trim) 输出 PWM"
        end
    end
```

以上调用链依据 `OttoDIY/OttoDIYLib`（`main` 分支）中的 `src/Otto.cpp`、`src/Oscillator.cpp`、`src/SerialCommand.cpp` 及 `examples/Otto_avoid/Otto_avoid.ino`、`examples/Otto_APP/Otto_APP.ino` 整理。

## 局限与风险

- 本页仅基于一篇第三方公众号文章，规格数字可能随版本变化。
- 仓库迁移、分叉和许可证以官方当前状态为准。

## 关联页面

- [humanoid-robot](./humanoid-robot.md)
- [open-source-humanoid-hardware](./open-source-humanoid-hardware.md)
- [机器人开源项目 18 例（微信 GitHubStore）索引](../overview/robot-open-source-wechat-githubstore-18-projects.md)
- [Locomotion](../tasks/locomotion.md)：以 4 舵机低成本双足结构为入门载体，实践基础步态生成与双足移动控制概念。
- [Parametric CAD Workflow · 参数化 CAD 建模流程](../methods/parametric-cad-workflow.md)：结合全 3D 打印结构件设计，用于定制和修改教育双足机器人的外壳与舵机支架模型。

## 推荐继续阅读

- [OttoDIY 组织](https://github.com/OttoDIY)

## 参考来源

- [wechat_githubstore_18_open_source_robot_projects.md](../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md)
