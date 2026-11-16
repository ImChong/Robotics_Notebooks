# raunaqbhirangi/anyskin（官方仓库核对笔记）

- **来源：** <https://github.com/raunaqbhirangi/anyskin>
- **开源许可：** MIT License（Python 库）；Arduino 固件头部注明 Beerware License
- **发布版本：** `v1.0.0`（支持 `pip install anyskin` 或源码可编辑安装）
- **核对日期：** 2026-11-14
- **核对范围：** `raunaqbhirangi/anyskin` 仓库 `README.md`、`LICENSE`、`setup.py`、`anyskin/sensor.py`、`anyskin/sensor_proc.py`、`anyskin/visualizations/anyskin_viz.py`、`arduino/5X_binary_burst_stream/5X_binary_burst_stream.ino`
- **通读：** 已完成

---

## 开源与仓库边界核查

- **已开源内容：** 完整的 Python 串口驱动库（`anyskin`）、5 颗磁力计实时可视化控制台工具（`anyskin_viz`）、Arduino 二进制突发流与文本流固件源码（`arduino/`）以及基础单元测试（`tests/sensor_test.py`、`sensor_proc_test.py`）。硬件模具 CAD 与制备文档通过项目页及配套 Google Drive 链接开放。
- **不包含的内容：** 本仓库定位为传感接口与数据流采集 SDK，**不包含**论文中的 LSTM 滑移检测训练管线与 BAKU 策略学习训练脚本。

---

## 源码架构与运行链路

1. **微控制器固件（`arduino/5X_binary_burst_stream/5X_binary_burst_stream.ino`）：**
   - 运行于 Adafruit QT Py 等兼容板，I2C 总线工作在 `400 kHz`，轮询同一条 I2C 总线上的 **5 颗 MLX90393** 三轴磁力计（地址 `0x0C, 0x0D, 0x0E, 0x0F, 0x10`），开启 `startBurst(0xF)` 连续突发采集模式；
   - 主循环 `loop()` 将 5 组 `MLX90393::txyz` 结构体（每颗芯片含温度 + `x, y, z` 共 4 个 `float32`，总计 20 个浮点数 = 80 字节）以二进制方式写入 115200 bps 串口并附加 `\r\n` 行尾标记（共 82 字节帧长）。
2. **同步采集底类（`anyskin/sensor.py`）：**
   - `AnySkinBase(serial.Serial)`：按 `4 * num_mags` 浮点数与 `4 * _msg_floats + 2` 字节帧长初始化串口，当 `temp_filtered=True` 时通过掩膜 `self._temp_mask[::4] = False` 过滤每颗芯片的首个温度字段，仅保留 15 维（5 芯片 × 3 轴）磁通数据；
   - `get_sample()` 检查串口缓冲积压（超过 4000 字节则重置并按 `\r\n` 重新帧同步），使用 `struct.unpack("@{}fcc")` 解析二进制负载并返回 `(timestamp, mag_xyz_array)`；另提供 `AnySkinDummy` 供无硬件离线测试。
3. **非阻塞多进程流与缓冲（`anyskin/sensor_proc.py`）：**
   - `AnySkinProcess(Process)` 在独立后台进程 `run()` 中托管 `AnySkinBase`，持续更新 `multiprocessing.Value`/`Array` 共享内存中的 `_last_time` 与 `_last_reading`（供实时策略高频读取最新一帧），并通过 `Pipe` 按 `chunk_size=10000` 批量回传录制缓存（`start_buffering()` / `get_buffer()`）。
4. **命令行实时可视化与零偏校准（`setup.py` + `anyskin/visualizations/anyskin_viz.py`）：**
   - 注册终端入口 `anyskin_viz = anyskin.visualizations.anyskin_viz:default_viz`；
   - `visualize()` 启动 `AnySkinProcess(num_mags=5, port=port)`，先采集 5 帧求均值得到静止零偏 `baseline`，随后以 **60 FPS** 在 `pygame` 窗口中叠加渲染各芯片位置的 Z 轴法向圆半径（正负用实心/空心区分）与旋转校正后的 XY 切向箭头，并支持按下键盘 **`B` 键**在线重新采集 5 帧刷新零位基准以抵消长期磁漂移，或传入 `-r` 保存录制文本日志。

---

## 映射到 Wiki

- [painode-146-anyskin.md](../../wiki/entities/painode-146-anyskin.md)
