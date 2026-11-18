---
type: entity
project_id: the5439workshop-actuator-2
project: https://github.com/the5439workshop/Actuator_2
code: https://github.com/the5439workshop/Actuator_2
tags: [hardware, actuator, planetary-gearbox, open-source, motor-control, foc, esp32, cnc, 3d-print]
status: complete
updated: 2026-11-18
original:
  - ../../sources/repos/the5439workshop_actuator_2.md
  - ../../sources/videos/why_walking_robots_are_so_hard_to_build_qKkivaZwqTo.md
related:
  - ../comparisons/open-source-qdd-actuator-projects.md
  - ./internal-cycloidal-actuator.md
  - ./ironless-qdd-actuator.md
  - ./opentorque-actuator.md
  - ./simplefoc.md
  - ../methods/joint-actuator-parameter-identification.md
  - ../methods/pid-control.md
  - ../queries/actuator-drive-chain-selection-loop.md
  - ../../roadmap/depth-torque-motor-design.md
sources:
  - ../../sources/repos/the5439workshop_actuator_2.md
  - ../../sources/videos/why_walking_robots_are_so_hard_to_build_qKkivaZwqTo.md
summary: "Actuator 2.0：the5439workshop 自研行星减速关节样机，公开 STL 机械件、PCB 制造资料与 ESP32-S3/SimpleFOC 示例驱动；强调制造公差、铝制齿轮寿命、散热与电压型力矩控制的验证边界。"
---

# Actuator 2.0（自研行星减速一体式机器人关节）

## 一句话定义

**Actuator 2.0** 是 [the5439workshop](https://github.com/the5439workshop/Actuator_2) 为自制行走机器人开发的关节执行器：将行星齿轮传动融入电机中空结构，搭配铝合金加工件、PETG 打印件、自研 PCB 和 ESP32-S3 / SimpleFOC 示例程序。项目公开设计文件，但作者明确说明**不是完整复刻教程**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CNC | Computer Numerical Control | 数控加工，影响齿轮及行星架的制造精度 |
| STL | Stereolithography | 已公开的机械网格模型格式 |
| PCB | Printed Circuit Board | 自制电机驱动板制造文件 |
| BOM | Bill of Materials | 公开的电子元件物料清单 |
| PETG | Polyethylene Terephthalate Glycol | 作者使用的部分 3D 打印结构材料 |
| FOC | Field-Oriented Control | SimpleFOC 的无刷电机控制方案 |
| SPI | Serial Peripheral Interface | 磁编码器与 ESP32-S3 之间的通信 |
| NTC | Negative Temperature Coefficient | 电机温度测量的热敏电阻 |
| PWM | Pulse-Width Modulation | 6 路功率级驱动控制信号 |

## 为什么重要

- 展示「**中空空间集成行星减速 + 自制电机/驱动板**」的个人工程实践，与 [OpenTorque 同步带传动](./opentorque-actuator.md)、[Internal Cycloidal 内嵌摆线](./internal-cycloidal-actuator.md) 构成可比较的实现路径。
- 开放的是 **STL + PCB/BOM + 示例固件**，有利于研究机械、电子、控制的系统集成；但缺少整机标定和可靠性数据。
- 来源视频描述了从 1.0 拉索/卷筒传动失效到 2.0 铝制行星减速结构的迭代。**已通读用户提供的完整视频英文文字稿**，可精确核对作者对机械尺寸、损耗口径和绕组模型的声明；仍未独立复测台架仪器。

## 核心结构与流程总览

仓库 `actuator_v2_stl/` 包含太阳轮、行星轮、齿圈、行星架、定转子适配件和外壳模型；官方 README 称齿轮选用 7075 铝，其余使用 6061 铝或 PETG。下面按**仓库可确认的模块职责**归纳设计和信号链，不将未验证的装配细节当作定论。

```mermaid
flowchart LR
  cmd["Serial 目标命令"] --> ctrl["ESP32-S3 + SimpleFOC"]
  enc["AS5047 SPI 磁编码器"] --> ctrl
  ctrl --> pwm["6PWM 驱动板"]
  pwm --> motor["定子绕组 + 永磁转子"]
  motor --> gear["太阳轮 / 行星轮 / 齿圈"]
  gear --> out["关节输出端"]
  ntc["NTC 温度采样"] --> ctrl
  ctrl --> fb["温度 / 角度串口反馈"]
```

**机械参数（作者视频自述）：** Actuator 2.0 采用 **6:1 行星减速**（太阳轮输入、齿圈固定、行星架输出），**900 g、直径 102 mm、厚 48 mm**；磁钢转子为 **21 对极**，与固件电机实例的极对数 21 一致。前代 Capstan 从 6:1 升至 7.5:1 后失效，切勿混淆。以上尺寸是作者自述，不是独立台架数据。

## 开源资产与控制实现

| 层 | 仓库路径 | 核对结论 |
|----|----------|----------|
| 结构/传动 | [actuator_v2_stl/](https://github.com/the5439workshop/Actuator_2/tree/main/actuator_v2_stl) | 太阳轮、行星轮、齿圈、载架等 STL 已发布，非公差合格证明 |
| PCB | [PCB/](https://github.com/the5439workshop/Actuator_2/tree/main/PCB) | Gerber、BOM、位置表和 PCB 预览可用 |
| 固件 | [code_driver/main.cpp](https://github.com/the5439workshop/Actuator_2/blob/main/code_driver/main.cpp) | SimpleFOC 控制逻辑及 Serial 交互已公开 |
| 构建 | [code_driver/platformio.ini](https://github.com/the5439workshop/Actuator_2/blob/main/code_driver/platformio.ini) | PlatformIO ESP32-S3 DevKitC，Arduino，Simple FOC 2.3.5 |
| 温度 | [TemperatureSensor.cpp](https://github.com/the5439workshop/Actuator_2/blob/main/code_driver/TemperatureSensor.cpp) | β 模型 NTC 温度换算 |

源码中 `motor.controller = MotionControlType::torque`，但 `motor.torque_controller = TorqueControlType::voltage`；这代表**电压型扭矩指令接口**，不是已核实的电流/输出扭矩闭环。`driver.voltage_power_supply = 20`、PWM 20 kHz、`motor.current_limit = 40` 为源码配置值，**不能代替实际供电、安全限流或动态力矩测试**。视频中作者称最终测试供电调至 **28 V**、另改部分参数，不能将 20 V 版本当作完整最终测试配置。

## 源码运行时序图

以下逐项对应官方 `code_driver/main.cpp` 的 `setup()` 和 `loop()`，仅表达已见的程序调用与数据交互。

```mermaid
sequenceDiagram
  autonumber
  participant Host as Serial 上位端
  participant Main as main.cpp
  participant Sensor as MagneticSensorSPI
  participant Motor as SimpleFOC BLDCMotor
  participant Driver as BLDCDriver6PWM
  participant Therm as TemperatureSensor
  Main->>Sensor: init()
  Main->>Driver: init()
  Main->>Motor: linkSensor(), linkDriver(), init(), initFOC()
  loop loop()
    Main->>Motor: loopFOC()
    Motor->>Sensor: 读取磁编码器角度
    opt 串口接收窗口，收到 T 数值及换行
      Host->>Main: Serial 命令
      Main->>Motor: 设置 target
    end
    Main->>Sensor: getAngle()
    Main->>Therm: readTemp(ADC 电压)
    Main-->>Host: Serial 打印温度,角度
    Main->>Motor: move()
    Motor->>Driver: 输出 PWM 电压控制
  end
```

**代码实际边界：** `Serial1.begin()` 仍被注释；即使代码切换 `RS485_DIR_PIN`，收发仍调用 `Serial`。风扇相关 GPIO 在 `setup()` 设为 LOW，`loop()` 中未见主动启停。**视频作者明确说最后将风扇直接接到电源**（四档调速设计效率欠佳），与源码未提供主动风扇调速相符；也不能据此推断已实现完整 RS485 或闭环温控。

## 工程实践：复现与验收建议

| 阶段 | 具体工作 |
|------|----------|
| 1. 审图 | 对照 STL、PCB BOM、机构装配关系及真实轴承规格；按设备加工能力重新确定齿轮/轴承公差 |
| 2. 电控 bring-up | 在限流台式电源下核验栅极、相序、编码器 SPI 与 NTC ADC；不得凭 `current_limit` 字段假设已有硬件保护 |
| 3. 控制验证 | 核验 SimpleFOC 电角度校准、命令输出与实际电流；必要时增加实际相电流测量 |
| 4. 台架 | 记录真实减速比、齿隙、输出扭矩—转速、连续温升、齿轮磨损和效率 |
| 5. 部署 | 补充超温/过流/失联保护、封闭防尘、风扇可靠性和总线实时性验证 |

## 视频原文核对：从失败到可靠性边界

### 机械与装配过程

- **Actuator 1.0：** 采用 3D 打印双折叠 Capstan 拉索传动，滑轮断裂；改金属滑轮仍难突破约 10 N·m。把减速比从 6:1 提到约 7.5:1 后，发生绳索打滑及转子输出轴断裂。
- **Actuator 2.0：** 设计转向铝合金主结构、PETG 辅助件，定子轴向长度加倍；定子中空区域集成固定齿圈、太阳轮输入、行星架输出的 **6:1 行星减速器**。
- **制造困难：** 作者用 7075 铝加工齿轮，受 1 mm 铣刀切深仅 3 mm 限制，叠两层齿轮；工业金属 3D 打印一体行星架在后加工时出现偏心/倾斜，最终拆成四块分别加工再组装。
- **电磁/冷却：** 三相星形绕组、21 对极磁钢转子、叠片磁路、磁编码器和 NTC 温度传感器。风扇改为直连供电，气流依次穿过驱动板、转子/定子间隙和行星齿轮。作者给出的尺寸为 900 g、直径 102 mm、厚度 48 mm。

### 堵转模型：匝数不等于免费扭矩

作者最重要的电气纠偏是：**电源侧输入功率、电机相电流、铜耗和输出力矩不是可以直接相互替代的量**。

- 在视频采用的理想堵转模型下，假设主要损耗是三相铜耗，表达式为 \(P_{\mathrm{cu}}=3I_{\mathrm{phase}}^2R_{\mathrm{phase}}\)（具体电流、电阻采用一致的定义）；未饱和时转矩近似与转矩电流成正比，因此固定绕组下 \(T\propto\sqrt{P_{\mathrm{cu}}}\)，而不是与铜耗线性相关。
- 作者对比 **18 匝/齿、两根并线、约 0.75 Ω** 与 **36 匝/齿、一根线、约 3 Ω** 的相电阻。铜量近似固定时，匝数约翻倍会使理想力矩常数约翻倍、相电阻约增四倍，同样铜耗下电流约减半，**堵转扭矩不因多绕线就自动提高**。
- PWM 功率级与绕组电感可实现电压/电流交换，母线输入电流不等于相电流，也不等同于 FOC 的转矩分量 Iq。
- **模型边界：** 绕组匝数仍影响速度常数、反电势、占空比和可用电压；磁饱和、铁耗、摩擦、温升、开关损耗及电阻温漂未被上述简化模型覆盖。

### 最终台架结果：损耗 ≠ 输入功率

| 作者视频报告 | 原话中的功率含义 | 尚不能推断 |
|-------------|----------------|-------------|
| 约 **10 N·m**，**30 W losses** | 30 W 指**损耗功率** | 输入功率、连续额定力矩和效率 |
| 接近 **20 N·m**，**100 W losses** | 100 W 指**损耗功率** | 输入功率、输出机械功率、稳态温升 |
| 最后将供电提高至 **28 V** | 避免占空比限制和过调制，作者还更改了其他参数 | 与此前测试可直接横向对比，或等同于公开固件默认 20 V 配置 |

这些是**视频作者的演示数据**，没有独立测功机校准、输出转速、持续时长、完整 TN 曲线和寿命试验。不能将接近 20 N·m 直接标注为「连续额定扭矩」。

## 局限与风险

- **制造一致性：** 作者明确提示文件按现状公开、没有装配教程，CNC 加工公差需自行调整。
- **材料/寿命：** 7075 铝制齿轮是加工条件下的妥协，作者不保证达到商用执行器寿命。
- **环境：** 穿流式冷却可能带入水和粉尘，作者不建议用于潮湿/多尘环境。
- **电气安全：** 示例采用电压型力矩控制；未见完整电流反馈、失联降级、驱动温控等可靠性验收。
- **开源边界：** STL、Gerber、BOM、代码可下载，但未发现根级 LICENSE 文件；复用和再发布前需确认许可。
- **证据边界：** 已通读用户提供的完整英文文字稿，准确记录作者自述，但未独立检验台架仪器、画面或寿命报告；内容完备并不等于产品成熟。

## 关联页面

- [开源 QDD / 力矩关节执行器项目对比](../comparisons/open-source-qdd-actuator-projects.md) — 与皮带、摆线及其他一体式方案比较开源粒度。
- [Internal Cycloidal Actuator](./internal-cycloidal-actuator.md) — 定子内部集成减速的相邻机械方案。
- [Ironless QDD Actuator](./ironless-qdd-actuator.md) — 自制绕组/齿轮组合的另一技术路径。
- [SimpleFOC](./simplefoc.md) — 驱动器固件实际依赖。
- [关节执行器参数辨识](../methods/joint-actuator-parameter-identification.md) — 转矩、电流、摩擦与惯量需实测。
- [PID Control](../methods/pid-control.md) — 调节稳定性不能仅凭设定参数推断。
- [执行器驱动链选型闭环](../queries/actuator-drive-chain-selection-loop.md) — 力矩、带宽、散热和总线联合选型。
- [力矩电机设计纵深路线](../../roadmap/depth-torque-motor-design.md) — 本项目适合 Stage 4 的驱动板集成与 Stage 6 的验收讨论。

## 参考来源

- [Actuator_2 官方仓库核对归档](../../sources/repos/the5439workshop_actuator_2.md)
- [Why Walking Robots Are So Hard to Build 视频全文归档](../../sources/videos/why_walking_robots_are_so_hard_to_build_qKkivaZwqTo.md)

## 推荐继续阅读

- [Actuator_2 官方仓库](https://github.com/the5439workshop/Actuator_2)
- [演示与制作过程视频](https://youtu.be/qKkivaZwqTo)
- [SimpleFOC 官方文档](https://docs.simplefoc.com/)
