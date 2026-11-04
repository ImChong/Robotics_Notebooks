---
type: entity
tags: [repo, radar, open-hardware, perception, uav, sdr, phased-array, abac-industry]
status: complete
updated: 2026-09-05
related:
  - ../queries/robot-perception-stack-selection-loop.md
  - ../overview/navigation-slam-autonomy-stack.md
  - ../overview/multirotor-simulation-planning-control-stack.md
  - ../overview/hub-state-estimation.md
  - ../entities/fast-lio.md
  - ../entities/px4-autopilot.md
  - ../queries/field-robotics-troubleshooting.md
  - ./project-quiver.md
sources:
  - ../../sources/repos/plfm_radar.md
summary: "AERIS-10（PLFM_RADAR）是开源 10.5 GHz 脉冲 LFM 相控阵雷达：3 km/20 km 双版本、FPGA 脉冲压缩与 CFAR、Python 地图 GUI，面向研究与无人机低成本主动测距实验。"
---

# AERIS-10（PLFM_RADAR）

**AERIS-10** 是 GitHub 上高关注度的 **开源脉冲线性调频（PLFM）相控阵雷达** 全栈：从原理图、PCB、FPGA 固件到 Python 可视化一应俱全，适合研究 **波束赋形、脉冲压缩、多普勒检测** 或评估 **主动雷达 vs LiDAR** 在野外机器人/无人机上的感知取舍。

| 字段 | 内容 |
|------|------|
| 机构 | 阿巴克工业（ABAC INDUSTRY） |
| 维护者 | Nawfal Motii |
| 许可 | 硬件 CERN-OHL-P；软件 MIT |

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PLFM | Pulse Linear Frequency Modulation | 脉冲线性调频，雷达 chirp 波形 |
| LFM | Linear Frequency Modulation | 线性调频，脉冲压缩常用波形 |
| CFAR | Constant False Alarm Rate | 恒虚警率检测，FPGA 片上实现 |
| MTI | Moving Target Indication | 动目标指示，抑制静止杂波 |
| FPGA | Field-Programmable Gate Array | 可编程门阵列，承担实时信号处理 |
| GPS | Global Positioning System | 全球定位，GUI 地图中心与点迹地理标注 |
| IMU | Inertial Measurement Unit | 惯性测量单元，俯仰/横滚修正目标坐标 |

## 为什么重要

- **成本与开放度**：相对商用相控阵雷达，提供可制造、可修改的 **10.5 GHz** 参考设计（硬件 CERN-OHL-P + 软件 MIT），降低 SDR/雷达教学与原型门槛。
- **机器人相关场景**：README 明确面向 **大学研究者、无人机创业团队、高级 maker**；可与 [PX4](../entities/px4-autopilot.md) 伴机或地面机器人导航实验组合，但 **不替代** [FAST-LIO](../entities/fast-lio.md) 等 LiDAR SLAM 栈——二者测距原理、角分辨率与 ROS 生态成熟度不同。
- **片上信号处理示范**：FPGA 完成 chirp 生成、脉冲压缩、多普勒 FFT、MTI、CFAR，是理解 **感知前端算力边界** 的硬件案例（对比纯 CPU/GPU 点云管线）。

## 核心结构/机制

| 子系统 | 关键器件 / 职责 |
|--------|------------------|
| 波形与 RF | DAC 生成 LFM chirp；LTC5552 上下变频；ADF4382 频率合成 |
| 波束赋形 | 4× ADAR1000 四通道相移器，16 元电子扫描 **±45°** |
| 收发前端 | 16× ADTR1107 LNA/PA；Extended 版 16× GaN 功放（约 10 W×16） |
| 实时处理 | XC7A50T FPGA：I/Q 下变频、抽取/滤波、脉冲压缩、多普勒、CFAR |
| 系统管理 | STM32F746：上电时序、外设、混合 AGC（FPGA/STM32/GUI） |
| 地理姿态 | UM982 GPS + GY-85 IMU + BMP180；点迹地图叠加 |
| 人机界面 | Python 3.8+ GUI，实时点迹与地图 |

### 双版本对照

| 参数 | AERIS-10N（Nexus） | AERIS-10X（Extended） |
|------|-------------------|----------------------|
| 最大距离 | 3 km | 20 km |
| 天线 | 8×16 贴片阵列 | 32×16 缝隙波导阵列 |
| 发射功率 | ~1 W × 16 | ~10 W × 16（GaN） |

## 流程总览

```mermaid
flowchart LR
  subgraph tx["发射链"]
    DAC["DAC\nLFM chirp"] --> MIX_TX["上变频"]
    PS["ADAR1000\n相移波束"] --> ANT["相控阵天线"]
    MIX_TX --> PS
  end
  subgraph rx["接收与检测"]
    ANT --> FE["ADTR1107\n前端"]
    FE --> FPGA["FPGA\n脉冲压缩 / 多普勒 / MTI / CFAR"]
  end
  subgraph host["上位与融合"]
    MCU["STM32\n时序 / 外设"]
    GPS["GPS + IMU\n姿态修正"]
    GUI["Python GUI\n地图点迹"]
    FPGA --> MCU
    MCU --> GUI
    GPS --> GUI
  end
```

## 常见误区或局限

- **不是即插即用 ROS 2 传感器**：无官方 `sensor_msgs`/`nav2` 桥接；接入 [导航栈总览](../overview/navigation-slam-autonomy-stack.md) 需自建驱动、时间同步与 `tf`。
- **Alpha 状态**：功能与文档仍在迭代；硬件组装需 RF/PCB 经验，Extended 版涉及 **高功率 RF** 安全与法规（频段、辐射合规需自行确认）。
- **与 LiDAR 分工不同**：雷达擅长 **远距离动目标与全天候测距**，角分辨率与稠密建图通常不如旋转 LiDAR；不宜直接等同 [LiDAR SLAM 选型](../comparisons/lidar-slam-lio-vio-selection.md) 中的 LIO 方案。
- **仿真缺口**：本库 [多旋翼仿真栈](../overview/multirotor-simulation-planning-control-stack.md) 以 AirSim/PyBullet 为主，**不含** 相控阵雷达物理仿真；算法验证多依赖本仓库自带仿真报告（`docs/`）或实测。

## 源码运行时序图

入口：`python 9_Firmware/9_3_GUI/GUI_V7_PyQt.py`（上位机）；MCU 入口 `int main(void)`（`9_Firmware/9_1_Microcontroller/9_1_3_C_Cpp_Code/main.cpp`）；FPGA 顶层 `radar_system_top`（`USB_MODE=1` 为 FT2232H 量产板分支）
```mermaid
sequenceDiagram
    autonumber
    participant U as 操作者
    participant G as GUI_V7_PyQt.py RadarDashboard
    participant R as radar_protocol.py FT2232HConnection RadarAcquisition
    participant F as FPGA radar_system_top
    participant M as MCU main.cpp
    participant A as ADAR1000_Manager 波束与收发切换

    U->>G: python GUI_V7_PyQt.py 执行 main 并创建 RadarDashboard
    G->>R: 打开 FT2232H 连接并启动 RadarDataWorker
    U->>G: 点击命令按钮 例如 MTI 开关 opcode 0x26
    G->>R: _send_fpga_cmd 经 RadarProtocol.build_command 生成 4 字节命令
    R->>F: FT2232HConnection.write 下发命令
    F->>F: usb_data_interface_ft2232h 解码为 cmd_opcode 等并写入寄存器

    M->>M: main 调用 systemPowerUpSequence 上电时序
    M->>M: 开机后 GPS_SendBinaryToGUI 经 CDC 发送一次 GPS 包
    M->>G: STM32 USB CDC 二进制 GPS 包
    G->>G: GPSDataWorker 读取 STM32USBInterface 并 parse_gps_data

    loop 每个雷达扫描帧 runRadarPulseSequence
        M->>A: setCustomBeamPattern16 下发 TX 与 RX 波束码
        M->>F: executeChirpSequence 翻转 GPIOD PIN8 通知新 chirp
        A->>A: pulseTXMode 与 pulseRXMode 切换收发
        F->>F: radar_transmitter 经 plfm_chirp_controller_enhanced 到 dac_interface_enhanced
        F->>F: radar_receiver_final 经 ad9484 到 ddc_400m 到 matched_filter_multi_segment
        F->>F: range_bin_decimator 到 mti_canceller 到 doppler_processor_optimized
        F->>F: cfar_ca 输出检测标志
        M->>F: 翻转 PIN9 与 PIN10 通知俯仰与方位步进
    end

    F->>R: 经 FT2232H 上传 0xAA 数据包与 0xBB 状态包
    R->>R: find_packet_boundaries 后 parse_data_packet
    R->>G: RadarDataWorker 取出帧并发出 frameReady
    G->>G: _on_frame_ready 更新距离多普勒与点迹显示
```

图依据：NawfalMotii79/PLFM_RADAR@62d4c55（默认分支 HEAD），实际读过 `9_Firmware/9_3_GUI/GUI_V7_PyQt.py`、`9_Firmware/9_3_GUI/v7/dashboard.py`（局部）、`9_Firmware/9_3_GUI/v7/workers.py`（局部）、`9_Firmware/9_3_GUI/v7/hardware.py`（头部与类定义）、`9_Firmware/9_3_GUI/radar_protocol.py`（类与函数签名、采集线程循环）、`9_Firmware/9_2_FPGA/radar_system_top.v`（例化与 USB_MODE 分支）、`9_Firmware/9_2_FPGA/radar_receiver_final.v`（例化列表）、`9_Firmware/9_2_FPGA/radar_transmitter.v`（例化列表）、`9_Firmware/9_2_FPGA/usb_data_interface_ft2232h.v`（头部注释与命令输出段）、`9_Firmware/9_1_Microcontroller/9_1_3_C_Cpp_Code/main.cpp`（main 与 runRadarPulseSequence、executeChirpSequence、GPS 发送片段）、`9_Firmware/9_1_Microcontroller/9_1_1_C_Cpp_Libraries/gps_handler.cpp`（GPS_SendBinaryToGUI）。未运行代码。

## 参考来源

- [sources/repos/plfm_radar.md](../../sources/repos/plfm_radar.md)
- [NawfalMotii79/PLFM_RADAR](https://github.com/NawfalMotii79/PLFM_RADAR)
- [项目文档（GitHub Pages）](https://NawfalMotii79.github.io/PLFM_RADAR/docs/)
- 源码核对（2026-11-02）：NawfalMotii79/PLFM_RADAR@62d4c55（默认分支 HEAD），读过 9_Firmware/9_3_GUI/GUI_V7_PyQt.py、v7/dashboard.py（局部）、v7/workers.py（局部）、v7/hardware.py（局部）、radar_protocol.py、9_Firmware/9_2_FPGA/radar_system_top.v、radar_receiver_final.v、radar_transmitter.v、usb_data_interface_ft2232h.v（局部）、9_Firmware/9_1_Microcontroller/9_1_3_C_Cpp_Code/main.cpp（局部）、9_1_1_C_Cpp_Libraries/gps_handler.cpp（局部）；时序图依据以上文件，未运行代码。

## 关联页面

- [导航·SLAM·自动驾驶开源栈总览](../overview/navigation-slam-autonomy-stack.md)
- [多旋翼仿真—规划—飞控开源栈总览](../overview/multirotor-simulation-planning-control-stack.md)
- [状态估计（知识链汇总）](../overview/hub-state-estimation.md)
- [野外机器人排障指南](../queries/field-robotics-troubleshooting.md)
- [Project Quiver](./project-quiver.md) — 作业级开源机架，高度计是商用 Nanoradar，不是本开源相控阵

## 推荐继续阅读

- [System Architecture（官方）](https://NawfalMotii79.github.io/PLFM_RADAR/docs/architecture.html)
- [Hardware Bring-Up Guide（官方）](https://NawfalMotii79.github.io/PLFM_RADAR/docs/bring-up.html)
