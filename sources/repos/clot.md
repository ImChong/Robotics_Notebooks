# CLOT（上交 / 上海 AI Lab 闭环全身遥操作）

> 来源归档

- **标题：** CLOT
- **类型：** repo
- **来源：** 上海交通大学 · 上海人工智能 Laboratory
- **链接：** <https://github.com/zhutengjie/CLOT>
- **项目页：** <https://zhutengjie.github.io/CLOT.github.io/>
- **论文：** <https://arxiv.org/abs/2602.15060>
- **入库日期：** 2026-06-12
- **一句话说明：** CLOT 官方实现：闭环全局运动跟踪遥操作管线（OptiTrack 采集 → 在线 IK 重定向 → Transformer+PPO 策略 → 全身 PD）；含 **Observation Pre-shift** 训练技巧、AMP 运动先验与 **20 h** 自采人体动作数据说明。
- **沉淀到 wiki：** [`wiki/entities/paper-amp-survey-16-clot.md`](../../wiki/entities/paper-amp-survey-16-clot.md)

---

## 核心定位

**CLOT**（*Closed-Loop Global Motion Tracking for Whole-Body Humanoid Teleoperation*）解决 **局部帧全身跟踪** 在长时程遥操作中不可避免的 **全局位姿漂移**，通过 **高频定位反馈** 在全局坐标系下闭环同步操作员与机器人。

---

## 论文承诺的发布内容

| 组件 | 说明 |
|------|------|
| 训练代码 | mjlab 仿真 + PPO + Transformer actor-critic |
| 部署管线 | OptiTrack 流式重定向 + 50 Hz 策略 + 400 Hz PD |
| 人体数据 | 约 20 h OptiTrack 采集（locomotion + 高动态全身） |
| 演示视频 | 项目页 Whole-Body / Robustness / Loco-Manipulation / Stability 分区 |

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-twist2](../../wiki/entities/paper-twist2.md) | 论文仿真对比基线：便携局部帧全身遥操作 |
| [teleoperation](../../wiki/tasks/teleoperation.md) | 长时程全身遥操作与数据飞轮 |
| [humanoid-amp-motion-prior-survey](../../wiki/overview/humanoid-amp-motion-prior-survey.md) | AMP 正则与 19 篇运动先验专题语境 |
| [pinocchio](./pinocchio.md) | 在线全身 IK 重定向引擎 |
