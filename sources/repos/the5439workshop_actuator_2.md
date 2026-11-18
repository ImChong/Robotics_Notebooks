# Actuator 2.0 — the5439workshop 自制腿足机器人执行器

> 官方仓库来源归档；仅归纳公开资料，非装配指导书。

- **标题：** Actuator_2 / Actuator 2.0
- **类型：** repo
- **作者：** the5439workshop
- **链接：** https://github.com/the5439workshop/Actuator_2
- **视频：** https://www.youtube.com/watch?v=qKkivaZwqTo
- **入库日期：** 2026-11-18
- **开源状态：** **部分开源（机械 STL、PCB 制造文件和示例驱动代码可获取；并非完整加工/标定/测试手册）**
- **许可核查：** 根目录未发现 LICENSE 文件；不能将可下载等同于获得明确再利用授权。
- **核对范围：** 官方 main 分支 readme.md 全文；PCB/、actuator_v2_stl/、code_driver/ 目录清单；code_driver/main.cpp、platformio.ini、TemperatureSensor.cpp 源码。二进制 Gerber/STL 文件未逐一进行几何/电气验证。
- **通读：** 已完成
- **沉淀到 wiki：** [Actuator 2.0](../../wiki/entities/the5439workshop-actuator-2.md)

## 官方资料说明

作者声明，这份公开资料是「按现状」提供的文件集合，不是制造教程；实际生产需要根据 CNC 精度调整公差。作者称齿轮使用 7075 铝，其他部件使用 6061 铝或 PETG 打印件；明确提示齿轮寿命不宜与商用品相提并论，且穿流式冷却可能带入粉尘/水分，项目不面向潮湿、多尘环境。

## 仓库交付内容

| 路径 | 已核实内容 | 不能据此推断 |
|------|------------|--------------|
| `actuator_v2_stl/` | `sun_gear.stl`、`planet_gear.stl`、`ring_gear.stl`、`carrier_*.stl`、`stator_*.stl`、`rotor_*.stl`、PETG 外壳等 | 没有公差检验报告、材料寿命测试 |
| `PCB/` | `BOM.csv`、`Gerber_2.zip`、贴片位置 CSV、驱动 PCB 预览图 | 不等于板级过流、热保护已被验证 |
| `code_driver/` | `main.cpp`、温度传感器类、`platformio.ini` | 不等于可稳定运行的整机网络驱动栈 |

仓库没有发现完整装配顺序、完整标定记录、可重复的扭矩—转速/热曲线及明确 LICENSE。

## 源码入口及实际逻辑

- 构建配置：`platformio.ini`，环境 `esp32-s3-devkitc-1-n8r2`，Arduino 框架，`askuric/Simple FOC@2.3.5`。
- `setup()` 初始化 AS5047 SPI 磁编码器、6 PWM 驱动、温度与串口，再执行 `motor.init()` / `motor.initFOC()`。
- 参数是 `driver.voltage_power_supply=20`、`driver.pwm_frequency=20000`、`motor.controller=torque`、`motor.torque_controller=voltage`；代码给出 `current_limit=40`，但未见实际采样电流闭环实现，不能据此确认 40 A 电流限制生效。
- `loop()` 调用 `motor.loopFOC()` 与 `motor.move()`；解析 `T<数值>` 串口命令设置 `motor.target`，回报 `温度,角度`。
- 代码中 `Serial1.begin(...)` 被注释，实际接收/发送使用 `Serial`；虽然切换 `RS485_DIR_PIN`，不能直接认定这已构成可用的 RS485 半双工驱动。
- 风扇 GPIO 被初始化为 LOW，未见在此文件内开启的控制逻辑；不能把机械散热设计等同于固件主动温控。
- NTC 温度估算位于 `TemperatureSensor.cpp`，采用 β 参数公式；当前代码需要额外测试 ADC 零/满量程、接线和异常输入行为。

## 与视频全文的核对

- **机械尺寸与减速：** 作者在完整视频文字稿中说明本代使用 **6:1** 内置行星减速器、样机 **900 g、直径 102 mm、厚 48 mm**。1.0 的 Capstan 传动才从 6:1 改到过约 7.5:1，随后绳索滑移和输出轴断裂。
- **磁钢极对数：** 作者描述转子安装 **21 对极**，与代码电机实例 BLDCMotor(21, 3.0) 一致。
- **绕组对照：** 作者把 **18 匝/齿、两根并线、约 0.75 Ω** 与 **36 匝/齿、单线、约 3 Ω** 进行比较；在相同铜量与堵转铜耗主导的简化模型下，匝数更多不保证同铜耗下扭矩更高。
- **最终测试：** 作者声称 **10 N·m / 30 W 损耗，接近 20 N·m / 100 W 损耗**。此处 W 是 *losses*，不是电源输入或机械输出功率。作者最后将供电改为 **28 V** 且调整其他参数，公开代码仍为 **20 V**。
- **风扇：** 视频称原四档调速器效率欠佳，最终风扇直接接电源；代码并没有主动的风扇调速路径。

## 待验证项

1. 视频文字稿的尺寸、减速比、力矩与损耗已经核对为作者自述，但仍需用 CAD、实物标定和独立台架数据验证。
2. 必须测机械啮合、轴承间隙、齿轮磨损及 7075 齿轮在负载下的寿命。
3. 上机前需补齐真实电流采样/保护、温升、编码器校准、失联保护、总线时延等测试。
4. 复用前核查许可与 Gerber、STL 的制造一致性。

## 对 wiki 的映射

- [Actuator 2.0 独立详情节点](../../wiki/entities/the5439workshop-actuator-2.md)
- [开源 QDD/力矩关节对比](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
- [力矩电机设计路线](../../roadmap/depth-torque-motor-design.md)
