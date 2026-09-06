# OPAL-RT — Hardware-in-the-Loop Testing

> 来源归档

- **标题：** Hardware-in-the-loop testing（OPAL-RT 产品页）
- **类型：** site / 厂商一手资料
- **URL：** https://www.opal-rt.com/hardware-in-the-loop/
- **厂商：** OPAL-RT Technologies
- **入库日期：** 2026-09-06
- **一句话说明：** 工业界 HIL 金标准叙述：用 **实时仿真植物模型** 替代物理被控对象，经 I/O 与 **被测设备（DUT）** 闭环，在安全可控条件下验证控制/保护/监测系统。
- **沉淀到 wiki：** 是 → [`wiki/concepts/hardware-in-the-loop.md`](../../wiki/concepts/hardware-in-the-loop.md)

---

## 核心机制（厂商定义）

1. **闭环结构：** 物理 plant 由运行在 HIL 仿真器上的 **实时模型** 替代；DUT（ECU、变频器、机器人控制器等）通过 **模拟与数字 I/O** 与仿真器交换信号。
2. **与传统台架对比：** 现场或功率实验室全系统测试保真度高，但 **昂贵、低效、有风险**；HIL 可测边缘工况与故障模式而不损坏真实设备。
3. **声称收益：** 样机出现前可完成约 **95%** 测试；可复用仿真器适配多项目；降低人员与设备安全风险。

## 典型应用行业（页面列举）

| 领域 | 用途 |
|------|------|
| 能源 / 电网 | 保护、稳定、可再生能源并网 |
| 电力电子 | 变流器、逆变器动态测试 |
| 汽车 | EV、BMS、辅助驾驶 ECU |
| 航空航天 | 飞控、航电、推进 |
| **自主系统 / 机器人** | 实时传感器仿真与 **co-simulation** |

## 与学术/机器人文献的交叉

- Fedák 等 2015 六轴工业臂 HIL 使用 **RT-LAB**（OPAL-RT 生态）作为主实时仿真执行环境 — 见 [fedak_2015_industrial_robot_6dof_hil_simulator.md](../papers/fedak_2015_industrial_robot_6dof_hil_simulator.md)。

## 对 wiki 的映射

- [Hardware-in-the-Loop](../../wiki/concepts/hardware-in-the-loop.md)
- [Software-in-the-Loop](../../wiki/concepts/software-in-the-loop.md) — 管线前段
