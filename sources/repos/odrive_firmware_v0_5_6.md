# ODrive Firmware fw-v0.5.6：经典开源运动控制栈

> 来源归档（官方一手源码与文档；版本锁定）

- **标题：** ODrive Firmware 0.5.6 (Legacy Open-Source Firmware)
- **类型：** repo
- **官方仓库：** https://github.com/odriverobotics/ODrive
- **代码：** https://github.com/odriverobotics/ODrive/tree/fw-v0.5.6 （**已开源**；已确认 tag 和文件可读取）
- **版本：** `fw-v0.5.6`；与本文其他固件版本严格区分
- **原文文档：** https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/docs/control.rst
- **入库日期：** 2026-11-19
- **核对范围：** 官方 `docs/control.rst`；`MotorControl/{main.cpp,axis.cpp,axis.hpp,controller.cpp,controller.hpp,motor.cpp,motor.hpp,encoder.cpp,acim_estimator.cpp,component.hpp,low_level.cpp}` 的关键接口、控制回调、端口连接、PWM 回调与安全相关实现；`communication/` 目录核查。定位重点是模块协作和运行时序，并非全仓逐行安全审计。
- **通读：** 已完成
- **一句话说明：** ODrive 0.5.6 以周期中断调度估计器/控制器/电机子模块，用端口连接设置反馈和力矩通路，另以 Axis 状态机组织校准与使能、失能。
- **对应 wiki：** [ODrive Firmware 架构](../../wiki/entities/odrive.md)

## 一手源码索引

| 源码 | 核实内容 |
|---|---|
| [main.cpp](https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/Firmware/MotorControl/main.cpp) | `rtos_main` 初始化接口/轴线程；`ODrive::control_loop_cb` reset port→checks/watchdog→all encoders→sensorless/controller/motor/current_control |
| [axis.cpp](https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/Firmware/MotorControl/axis.cpp) | `Axis::start_closed_loop_control` 的传感器端口、力矩输出、电流命令和电角度接线；`run_state_machine_loop` 和 `stop_closed_loop_control` |
| [controller.cpp](https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/Firmware/MotorControl/controller.cpp) / [controller.hpp](https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/Firmware/MotorControl/controller.hpp) | 输入位置/速度/力矩、多种 input mode、反馈端口、`torque_output_`、反齿槽与保护相关处理 |
| [motor.cpp](https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/Firmware/MotorControl/motor.cpp) / [motor.hpp](https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/Firmware/MotorControl/motor.hpp) | `Motor::update` 力矩→dq 电流设定/限幅；`current_meas_cb`、`pwm_update_cb`、`disarm`、电流环增益 |
| [component.hpp](https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/Firmware/MotorControl/component.hpp) | `InputPort::connect_to`；`OutputPort::present/previous/any/reset` 的数据新鲜度合同 |
| [encoder.cpp](https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/Firmware/MotorControl/encoder.cpp) | 位置/速度/相位估计、传感器初始化及校准 |
| [acim_estimator.cpp](https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/Firmware/MotorControl/acim_estimator.cpp) | 交流感应电机估计器是独立组件，而非 Motor 里的唯一实现 |
| [docs/control.rst](https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/docs/control.rst) | 位置 P/速度 PI/电流 PI 级联；文档明确说明 current controller 在 Motor 而非 Controller |

## 实际控制路径

- `rtos_main` 完成 `init_communication`、功率驱动/编码器初始化、`start_adc_pwm`，然后为每轴启动状态机线程。
- 轴进入闭环：`start_closed_loop_control` 连接反馈端口、`controller_.torque_output_`→`motor_.torque_setpoint_src_`；将电角度/角速度连接到 current controller，并 `motor_.arm`。
- 周期回调：`ODrive::control_loop_cb` 清除旧端口输出的“本拍有效”标志、检查端点/看门狗、更新所有轴的 Encoder，再依次执行每轴 sensorless estimator、Controller、OpenLoopController、Motor、current_control。
- ADC/PWM 回调：`Motor::current_meas_cb` 处理电流采样与快速安全检查；`Motor::pwm_update_cb` 根据控制律输出 PWM compare，异常时调用失能路径。
- **设计细节**：`present` 只允许取本拍生产值；`previous` 取上一拍值。执行顺序和端口时效性必须配合，不能简单认为任意节点随时可交换。

## 代码与文章差异

- `uart_poll()` 在 `ODrive::control_loop_cb` 中出现；“命令与实时完全隔离”应理解为**职责分层**，不是绝对的代码执行隔离。
- `Motor::update` 的 ACIM 分支处理励磁及磁链转矩换算，还调用估计器；`ACIMEstimator` 本身是独立模块，不能据此说 ACIM 全部放进 Motor。
- 源码在特定硬件/固件场景下实现多轴/多接口；不能推断成后续所有产品都有相同 API、硬件、许可证和运行限制。

## 工程意义与注意事项

- 适合研究运动控制底软如何隔离命令、运动级目标、功率变换与状态机。
- 不能把这套遗留固件的 PWM 调度频率、接口支持和安全机制无条件用于新 ODrive 商用产品。
- 上机之前应独立验证限流、再生制动、故障门控、校准顺序与功率级互锁。源码核对不等于任何安全认证。
