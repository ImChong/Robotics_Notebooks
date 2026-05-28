# open_duck_mini_runtime

> 来源归档

- **标题：** Open Duck Mini Runtime
- **类型：** repo
- **来源：** apirrone（Open Duck Project）
- **链接：** https://github.com/apirrone/Open_Duck_Mini_Runtime
- **入库日期：** 2026-05-28
- **一句话说明：** 在 **Raspberry Pi Zero 2W** 上运行 Open Duck Mini v2 策略的机载 Runtime：Feetech 舵机 I2C 控制、IMU、Xbox 手柄蓝牙、ONNX 策略推理与 sim2real 上机 checklist。
- **沉淀到 wiki：** 是 → [`wiki/entities/open-duck-mini-runtime.md`](../../wiki/entities/open-duck-mini-runtime.md)

---

## 硬件平台

| 组件 | 说明 |
|------|------|
| 主控 | Raspberry Pi Zero 2W（64-bit Raspberry Pi OS Lite） |
| 执行器 | Feetech 总线舵机（与 [Open_Duck_Mini](open_duck_mini.md) BOM 一致） |
| 传感 | IMU（`scripts/imu_test.py`） |
| 输入 | Xbox One 手柄（Bluetooth：`bluetoothctl` pair/trust/connect） |
| 音频 | Adafruit MAX98357 I2S 功放（可选，见 Adafruit 教程） |

## 软件栈

- Python 3 + **virtualenvwrapper**（`open-duck-mini-runtime` 环境）
- `pip install -e .`（**v2** 分支为当前部署主线）
- 关键脚本：
  - `scripts/v2_rl_walk_mujoco.py` — ONNX 策略 MuJoCo 预演 / 上机前验证
  - `scripts/find_soft_offsets.py` — 关节软限位偏置标定 → 写入 `hwi_feetech_pwm_control.py`
  - `scripts/test_xbox_controller.py` — 手柄测试

## 系统配置要点

- **I2C** 启用（`raspi-config`）；可选 400 kHz
- **USB 串口延迟：** `/etc/udev/rules.d/99-usb-serial.rules` 设置 `latency_timer=1`（FTDI）
- **电机控制板 udev 规则：** README 标注 TODO
- 部署前需完成 [checklist.md](https://github.com/apirrone/Open_Duck_Mini_Runtime/blob/v2/checklist.md)

## 与训练仓的衔接

1. [Open_Duck_Playground](open_duck_playground.md) 训练并 `export_onnx`
2. 将 ONNX 拷至 Pi（或 README 提供的预训练 `BEST_WALK_ONNX*.onnx`）
3. 完成关节偏置标定与 checklist
4. 真机运行 walking 策略

## 与本仓库 wiki 的映射

- 实体页：`wiki/entities/open-duck-mini-runtime.md`
- 交叉：[sim2real.md](../../wiki/concepts/sim2real.md)、[processor-in-the-loop-sim2real.md](../../wiki/concepts/processor-in-the-loop-sim2real.md)（低算力机载推理边界）
