# TWIST2（Amazon FAR 全身遥操作与数据采集）

> 来源归档

- **标题：** TWIST2
- **类型：** repo
- **来源：** Amazon FAR（Frontier AI & Robotics）
- **链接：** <https://github.com/amazon-far/TWIST2>
- **项目页：** <https://yanjieze.com/projects/TWIST2/>
- **论文：** <https://arxiv.org/abs/2511.02832>
- **入库日期：** 2026-06-12
- **一句话说明：** TWIST2 官方仓库：**训练/部署代码**、**控制器 checkpoint**、**2-DoF 颈硬件设计（BOM/3D 打印/装配视频）** 全开源，支持在 **Unitree G1** 上复现便携全身遥操作与 visuomotor 自主策略管线。
- **沉淀到 wiki：** [`wiki/entities/paper-twist2.md`](../../wiki/entities/paper-twist2.md)

---

## 核心定位

**TWIST2** 是 [*Scalable, Portable, and Holistic Humanoid Data Collection System*](https://arxiv.org/abs/2511.02832) 的官方实现入口，与 [项目页](https://yanjieze.com/projects/TWIST2/) 及 [TWIST-Data 社区数据集](https://twist-data.github.io/) 配套。

---

## 仓库承诺的发布内容（项目页 / README）

| 组件 | 说明 |
|------|------|
| 颈增广硬件 | 2-DoF 颈 BOM、3D 打印件、装配教程 |
| 遥操作栈 | PICO 4 Ultra + Motion Trackers + XRoboToolkit 配置与全身流 |
| 低层控制器 | sim2real RL 全身 motion tracking 策略与 checkpoint |
| 高层自主 | visuomotor 策略（如 Diffusion Policy）训练与部署脚本 |
| 开源数据 | 全身 loco-manipulation 示范（twist-data.github.io） |

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-twist](../../wiki/entities/paper-twist.md) | 前作：全身模仿遥操作系统与跟踪控制器基座 |
| [paper-bifrost-umi](../../wiki/entities/paper-bifrost-umi.md) | 对照：无机器人 UMI 式采集 vs TWIST2 真机便携遥操作 |
| [paper-amp-survey-16-clot](../../wiki/entities/paper-amp-survey-16-clot.md) | CLOT 仿真对比基线之一（全局闭环 vs 局部帧跟踪） |
| [teleoperation](../../wiki/tasks/teleoperation.md) | 人形全身遥操作任务语境 |
