# fpv-labs/ego-oscar — Ego-OSCAR 开源采集固件及主机录制器

> 官方代码与硬件复现入口归档（2026-11-17 核对）

- **官方仓库：** <https://github.com/fpv-labs/ego-oscar>
- **发布说明：** <https://www.fpvlabs.ai/ego-oscar>
- **对应论文：** [Ego-OSCAR（arXiv:2608.08285v3）](https://arxiv.org/abs/2608.08285)，2026-10-28 修订
- **对应数据：** <https://huggingface.co/datasets/fpvlabs/stereo-550>（gated + FPV Labs 数据许可）
- **复核日期：** 2026-11-17
- **类型：** repo / open-hardware / capture stack
- **许可：** 代码仓 `Apache-2.0`；不得推导数据集也适用此许可
- **一句话说明：** 官方已发布 ESP32-S3 的 ICM-20948 + VSYNC + 看门狗固件、Radxa 上的 Rockchip FFmpeg 双目录制器及帧同步核验脚本，可按硬件配置启动录制。

## 源码入口核对

| 路径 | 代码作用及使用入口 |
|---|---|
| [README.md](https://github.com/fpv-labs/ego-oscar/blob/main/README.md) | 安装依赖与固件烧录说明；运行 `python3 radxa/fpv_recorder.py` |
| [firmware/esp32_coprocessor_led_watchdog/esp32_coprocessor_led_watchdog.ino](https://github.com/fpv-labs/ego-oscar/blob/main/firmware/esp32_coprocessor_led_watchdog/esp32_coprocessor_led_watchdog.ino) | Arduino IDE 烧录至 XIAO ESP32-S3；采集 IMU、计 VSYNC，串口输出 `G0` 和中断统计 `I0`，处理 W 状态命令 |
| [radxa/fpv_recorder.py](https://github.com/fpv-labs/ego-oscar/blob/main/radxa/fpv_recorder.py) | `main` 打开串口、判断摄像头；`main_recording_mode` GPIO 按钮触发；`start_recording` 启动 IMU/中断与 FFmpeg；`stop_recording` 停止并保存会话 |
| [radxa/README.md](https://github.com/fpv-labs/ego-oscar/blob/main/radxa/README.md) | 主机依赖、默认硬件接口、输出约定 |
| [radxa/check_fsn.py](https://github.com/fpv-labs/ego-oscar/blob/main/radxa/check_fsn.py) | 用 `python3 radxa/check_fsn.py path/to/session_imu.csv` 检查 FSN 序列 |
| [radxa/imu_test.py](https://github.com/fpv-labs/ego-oscar/blob/main/radxa/imu_test.py) | LSM9DS1 I2C 旧/备选硬件诊断；不是论文 ICM-20948 默认设备规格 |

## 实际采集链路

1. XIAO ESP32-S3 监视相机 VSYNC、以配置频率采集 IMU；UART 发送十六进制 `G0`、`I0` 消息给 Radxa，并用心跳保持看门狗与 LED 状态。
2. Radxa 运行 `fpv_recorder.py`，长按 GPIO 按钮后发 `W101`、`I100`、`G100`；用 V4L2 从 `/dev/video0` 读取 2560×720 左右拼接 MJPEG 图像。
3. `subprocess.Popen` 调用 Rockchip FFmpeg，`mjpeg_rkmpp` 硬解，`h264_rkmpp` 硬编，按 300 秒输出 MP4；ESP 串口异步落盘 IMU CSV 与帧同步信息。
4. 停录时结束 FFmpeg、停止 IMU 和计数、写入 session 元数据与 clip 偏移；`check_fsn.py` 可做录制后帧序号连续性检查。

## 工程边界与来源差异

- 运行前需要 Rockchip MPP FFmpeg；官方仓不存储第三方预编译二进制。录制器默认 `/dev/ttyS6`、921600 baud、`/dev/video0`、`gpiochip1` line 8，需与真实设备接线核对。
- `radxa/fpv_recorder.py` 默认 `ESP_IMU_RATE_HZ = 180`，arXiv v3 和 HF 卡约 180 Hz，官网发布博客还写 120 Hz。不能在未实测时说各版本硬件都稳定工作在相同采样率。
- 代码仓是采集栈；没有从这部分源码推断端到端策略训练能力，也没有将录制器输出目录等同于 HF 发布集的整理后目录。
- 数据集需 gated 申请；开源代码不代表数据许可无限制。

## 对 wiki 的映射

- [Ego-OSCAR 论文实体](../../wiki/entities/paper-ego-oscar.md)
- [论文全文摘录与通读记录](../papers/ego_oscar_arxiv_2608_08285.md)
- [官方项目页归档](../sites/fpvlabs-ego-oscar.md)
- [Stereo-550 数据集归档](../datasets/stereo-550.md)
