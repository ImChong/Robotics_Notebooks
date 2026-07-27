# HandUMI Hardware（robonet-ai/handumi-hw）

> 来源归档

- **标题：** HandUMI Hardware
- **类型：** repo（可穿戴无机器人示教硬件）
- **链接：** https://github.com/robonet-ai/handumi-hw
- **旧链接（301）：** https://github.com/BrikHMP18/HandUMI → 迁入本仓
- **软件仓：** https://github.com/robonet-ai/handumi-sw
- **文档站：** https://robonet-ai.github.io/handumi-sw/
- **机构：** RoboNet AI（项目主导 / 原硬件设计：[BrikHMP18](https://github.com/BrikHMP18)）
- **许可证：** Apache-2.0
- **入库日期：** 2026-07-27
- **一句话说明：** 面向**平行夹爪机械臂**的手持/可穿戴 UMI 变体：拇指–食指自然 pinch 开合、可更换夹爪 tip、约 **$110** 零件成本；无机器人在环采集双臂示范，再交 [handumi-sw](./handumi-sw.md) 重定向。
- **沉淀到 wiki：** [handumi](../../wiki/entities/handumi.md)

---

## 核心定位

传统 leader–follower 双臂采集至少需要 **两套 follower + 两套 leader**（或 VR 充当 leader），成本高且受「臂必须运到采集现场」约束。HandUMI 把采集接口搬到操作者手上：

- 机身 / 相机座 / 舵机 / 追踪器安装保持不变；
- **仅更换可拆卸夹爪 tip** 即可对齐不同平行夹爪机器人；
- 当前 tip 目标：AgileX PiPER、ARX X5 2023、Dream Gripper（TRLC）、Trossen WidowX AI、原版 UMI 夹爪；可比几何的平行夹爪可自设计 tip。

---

## 记录信号（README）

| 通道 | 来源 |
|------|------|
| SE(3) 腕部位姿 | PICO 4 Ultra / Meta Quest 3 头显世界系 + 双手柄 |
| 夹爪开合宽度 | **Feetech 舵机编码器直测**（非 fiducial / 分割间接估计） |
| 腕部视角视频 | 机载鱼眼 USB（UVC）相机 |

---

## 对 wiki 的映射

- 实体页：[handumi](../../wiki/entities/handumi.md)
- 软件归档：[handumi-sw](./handumi-sw.md)
- 对照：[ALOHA](../../wiki/entities/aloha.md)、[BifrostUMI](../../wiki/entities/paper-bifrost-umi.md)、[mimic U1](../../wiki/entities/mimic-wearable-u1.md)
