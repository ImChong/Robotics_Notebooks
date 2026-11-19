# ODrive v3.x Firmware 0.5.6：双轴实时控制一手源码核对

> 来源归档（官方 GitHub release tag 与核心实现代码）

- **名称：** ODrive / Firmware v0.5.6（v3.x）
- **类型：** repo / firmware
- **官方仓库：** https://github.com/odriverobotics/ODrive
- **锁定版本：** [fw-v0.5.6](https://github.com/odriverobotics/ODrive/tree/fw-v0.5.6)
- **来源日期：** 以官方 tag 为准（仓库 CHANGELOG 的 tag 内容将 0.5.6 标记为 Unreleased；不要把标签内容视为正式发布日期）
- **入库日期：** 2026-11-19
- **开源状态：** **已开源**，可读取 v3.x 固件源码；该公开仓库明确不覆盖新一代 ODrive 产品线的闭源固件
- **核对范围：** 在 `fw-v0.5.6` 逐段阅读 `Firmware/Board/v3/board.cpp` 的 TIM8 更新/控制环两个中断处理函数；`Firmware/Board/v3/Inc/main.h` 与 `board.h` 的时钟/周期定义；`Firmware/MotorControl/main.cpp` 的 `sampling_cb()`、`control_loop_cb()` 与 `osKernelStart()`；`axis.cpp` 的 `start_thread()`/等待信号；`controller.cpp` 的 `move_to_pos()`、`INPUT_MODE_TRAP_TRAJ`、`INPUT_MODE_PASSTHROUGH` 和 `INPUT_MODE_POS_FILTER`；`trapTraj.cpp` 的 `planTrapezoidal()` 与 `eval()`；官方 `CHANGELOG.md` 的 0.5.x 更新记录
- **通读：** 已完成
- **执行/硬件验证：** 未在 MCU 编译、烧录或测量 WCET，本次结论是静态源码核对。

## 固定版本与关键入口

| 文件 | 已核对的入口 | 源码所示 |
|---|---|---|
| [Board/v3/board.cpp](https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/Firmware/Board/v3/board.cpp) | `TIM8_UP_TIM13_IRQHandler()` | 依方向区分采样/更新；上数时执行 `sampling_cb()`，通过 `NVIC->STIR` 触发 ControlLoop；下数时 PWM 先置 50% 占空比 |
| 同上 | `ControlLoop_IRQHandler()` | ADC 完成校验 → M0/M1 `current_meas_cb` → `odrv.control_loop_cb` → DC 偏置校准 → M0/M1 `pwm_update_cb` → deadline 校验 |
| [MotorControl/main.cpp](https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/Firmware/MotorControl/main.cpp) | `sampling_cb()` | 两轴编码器 `sample_now()`，注释禁止在高优先级采样 ISR 中调用 FreeRTOS |
| 同上 | `control_loop_cb()` | 两轴依次更新编码器/估计器/运动控制与电流控制；末尾对 Axis 线程 `osSignalSet` |
| 同上 | `main()` | USB/UART 事件队列、信号量、RTOS 主线程创建及 `osKernelStart()` |
| [MotorControl/axis.cpp](https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/Firmware/MotorControl/axis.cpp) | `Axis::start_thread()` | 用 CMSIS `osThreadCreate()` 启动 `run_state_machine_loop()`；该任务不是高速 FOC 调度器 |
| [MotorControl/controller.cpp](https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/Firmware/MotorControl/controller.cpp) | `Controller::move_to_pos()`、`Controller::update()` | 对当前单目标 `planTrapezoidal()`，新 `input_pos` 使当前轨迹重新规划；可选位置直通/滤波模式 |
| [MotorControl/trapTraj.cpp](https://github.com/odriverobotics/ODrive/blob/fw-v0.5.6/Firmware/MotorControl/trapTraj.cpp) | `planTrapezoidal()`、`eval()` | 梯形/三角速度规划，终点位置为目标，终点速度为 0 |

## 时间预算（源码公式推得，非台架实测）

- `TIM_1_8_CLOCK_HZ = 168000000`；`TIM_1_8_PERIOD_CLOCKS = 3500`；`TIM_1_8_RCR = 2`。
- `CURRENT_MEAS_PERIOD = 2 × 3500 × (2+1) / 168000000 = 0.000125 s`，即 **125 μs / 8 kHz**；控制周期相当于 **21,000 个 168 MHz 定时器 tick**。
- 中心对齐 PWM 基本载波频率是 `168 MHz / (2 × 3500) = 24 kHz`，与每 125 μs 执行一次控制回调不同。
- `TIM8_UP_TIM13_IRQHandler` 和 `ControlLoop_IRQHandler` 的优先级配置需结合板级 NVIC 初始化核对；“RTOS 线程保证 8 kHz”不是该实现。
- `ERROR_TIMER_UPDATE_MISSED`、`ERROR_BAD_TIMING`、`ERROR_CONTROL_DEADLINE_MISSED` 在所核对 `board.cpp` 路径都同时对 M0、M1 执行 `disarm_with_error`。
- 静态可算的是 deadline，不可推断真实最坏执行时间（WCET）；应在真实双轴工况测量 profiling 与 deadline slack。

## 需要避免的泛化

1. `fw-v0.5.6` 的 ODrive v3.x 架构不代表后续 Pro / S1 / Micro 固件。
2. “两个 FOC 在同一个 ISR 顺序执行”指回调链顺序，并非两个独立 RTOS Task；双轴硬件 PWM 仍各自按定时器工作。
3. 固件存在 `INPUT_MODE_PASSTHROUGH` 与 `INPUT_MODE_POS_FILTER` 进行外部连续设定值更新，但软件上位机自身的通讯延迟、丢帧、队列与同步需要独立处理。
4. 本次没做真机负载与中断抖动测量；不声明可保证具体余量或闭环稳定性。

## 对 wiki 的映射

- [ODrive v3 Firmware 0.5.6：FreeRTOS 与双轴硬实时控制](../../wiki/entities/odrive-v3-firmware-rtos-foc.md)
