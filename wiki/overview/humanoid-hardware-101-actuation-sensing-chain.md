---
type: overview
tags: [humanoid, hardware, motor, gearbox, encoder, qdd, harmonic, category-hub]
status: complete
updated: 2026-06-01
summary: "Humanoid Hardware 101 · 02 传动与感知链 — 外转子伺服+有传感器换相、谐波/RV/行星减速器路线之争、双编码器与磁编绝对式；QDD 利 RL 扭矩透明。"
related:
  - ./humanoid-hardware-101-technology-map.md
  - ./humanoid-hardware-101-linear-transmission-bearings.md
  - ./humanoid-hardware-101-integrated-actuators.md
  - ../overview/motor-drive-firmware-bus-protocols.md
  - ../methods/amp-reward.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_hardware_101.md
  - ../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md
---

# Humanoid Hardware 101 · 02：传动与感知链

> **图谱分类节点**：**电机、减速器、编码器** 三章合并；关节「转什么、减多少、知在哪」的部件层，集成见 [04 集成执行器](./humanoid-hardware-101-integrated-actuators.md)。

## 电机（核心结论）

- 关节主流：**永磁无刷直流伺服**，**有传感器电子换相**、**外转子**、**径向磁通**、**钕永磁**。
- **伺服** = 编码器反馈的位置/速度/力矩闭环；步进几乎不用于人形关节。
- **灵巧手**：**空心杯**（无铁芯、低齿槽、散热差）。

## 减速器（核心结论）

| 类型 | 特点 | 人形典型部位 |
|------|------|--------------|
| **谐波** | 紧凑、低回程间隙、效率较低、难反向驱动 | 肩肘腕等精密关节 |
| **摆线/RV** | 抗冲击、大接触面；纳博特斯克等主导 | 腰髋膝等大关节 |
| **行星** | 可反向驱动、扭矩透明性较好 | QDD 路线 |

**行业争论：** 传统「微米精度 + 十年寿命」vs 整机厂把减速器当 **12–18 个月可换消耗件**；后者利好 **80% 性能、更快迭代** 的中国供应商（文内以利得传动等为例）。

**QDD vs 高减速比（RL 视角）：**

| 特性 | QDD（行星等） | 高减速比（谐波/RV） |
|------|---------------|---------------------|
| 扭矩透明 | 高，电流≈输出力矩 | 低，摩擦屏蔽外力 |
| 反向驱动 | 好 | 差 |
| RL 适配 | 动态探索、接触学习 | 高载精密、静态任务 |

关节载荷需区分 **径向 / 轴向 / 弯矩**；**交叉滚子轴承** 商品化减轻多轴承堆叠。

## 编码器（核心结论）

- **电机轴 + 关节输出端双编码器**：弥补间隙、柔性、打滑，缓解 sim2real 差距。
- 技术：**磁编** 成足式/人形默认（耐污、紧凑）；光学高分辨率但怕尘振。
- **绝对式** 关节必备（掉电知位、安全恢复）；增量式多用于转速监测。

## 关联页面

- [Humanoid Hardware 101 技术地图](./humanoid-hardware-101-technology-map.md)
- [直线传动与轴承](./humanoid-hardware-101-linear-transmission-bearings.md)
- [产业与成本地缘](./humanoid-hardware-101-supply-chain-economics.md)
- [电机驱动与总线协议](./motor-drive-firmware-bus-protocols.md)

## 参考来源

- [wechat_human_five_humanoid_hardware_101.md](../../sources/blogs/wechat_human_five_humanoid_hardware_101.md)
- [wechat_humanoid_hardware_101_2026-06-01.md](../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md)
