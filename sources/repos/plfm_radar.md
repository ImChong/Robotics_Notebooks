# AERIS-10 (PLFM_RADAR)

> 来源归档

- **标题：** AERIS-10 — Open Source Pulse LFM Phased Array Radar
- **类型：** repo / open-hardware
- **链接：** https://github.com/NawfalMotii79/PLFM_RADAR
- **Stars：** ~22k（2026-07）
- **入库日期：** 2026-07-08
- **一句话说明：** 开源、低成本 10.5 GHz 脉冲线性调频（PLFM）相控阵雷达：双版本（3 km / 20 km）、FPGA 片上脉冲压缩与 CFAR、Python 地图 GUI，面向研究者与无人机开发者。
- **沉淀到 wiki：** [aeris-10-plfm-radar](../../wiki/entities/aeris-10-plfm-radar.md)、[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)、[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位

**AERIS-10**（仓库名 PLFM_RADAR）旨在 **民主化相控阵雷达**：完整开放硬件（CERN-OHL-P）与软件（MIT），模块化可 hack，覆盖波形生成、波束赋形、脉冲压缩、多普勒与目标跟踪全链路。

| 版本 | 代号 | 最大距离 | 天线 |
|------|------|----------|------|
| Nexus | AERIS-10N | 3 km | 8×16 贴片阵列 |
| Extended | AERIS-10X | 20 km | 32×16 介质填充缝隙波导阵列 |

共同参数：**10.5 GHz**、电子波束扫描 **±45°**（方位/俯仰）、机械 360° 步进电机扫描、**FPGA + STM32** 异构处理。

---

## 硬件子系统（摘要）

- **电源管理板** — 多路电压时序与滤波
- **频率合成板** — AD9523-1 低抖动时钟 → ADF4382 RX/TX 合成器、DAC/ADC/FPGA
- **主控板** — DAC 生成 LFM chirp；LTC5552 上下变频；4× ADAR1000 相移器（16 通道波束赋形）；16× ADTR1107 收发前端；**XC7A50T FPGA** 做基带与检测；**STM32F746** 做上电时序、外设与 GPS/IMU
- **功放板（Extended）** — 16× QPA2962 GaN，约 10 W×16
- **外围** — UM982 GPS、GY-85 IMU、BMP180 气压计、散热与滑环

---

## 信号处理流水线

1. DAC 生成 LFM chirp → 上变频发射  
2. ADAR1000 相移 → 电子波束指向  
3. FPGA：ADC 采集 → I/Q 下变频 → 抽取/滤波 → 脉冲压缩 → 多普勒 FFT → MTI → CFAR  
4. STM32：电源/外设/GPS-IMU 姿态修正  
5. Python GUI：实时点迹、地图叠加、雷达控制  

文档站点：<https://NawfalMotii79.github.io/PLFM_RADAR/docs/>

---

## 对 wiki 的映射

- 实体页：[aeris-10-plfm-radar](../../wiki/entities/aeris-10-plfm-radar.md)
- 导航栈对照（LiDAR/VIO 之外的 **主动测距** 硬件参考）：[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)
- 多旋翼栈（README 明确面向 **drone developers**）：[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)
- 状态估计专题（雷达测距与 GPS/IMU 融合）：[topic-state-estimation](../../wiki/overview/topic-state-estimation.md)

---

## 维护者备注

- **状态：** Alpha，部分功能仍在 issue 中跟踪；非 ROS 2 原生节点，接入 Nav2/PX4 需自建驱动与坐标系桥接。
- **许可：** 硬件 CERN-OHL-P，软件 MIT；社区推动由纯 MIT 改为硬件专用许可以覆盖 RF 产品责任。
- **机构：** Nawfal Motii / ABAC INDUSTRY（摩洛哥）
