---
type: entity
tags: [manipulation, end-effector, gripper, hardware, open-source, dynamixel, westwood-robotics]
status: complete
updated: 2026-06-30
related:
  - ../tasks/manipulation.md
  - ../overview/topic-grasp.md
  - ./orca-hand.md
  - ./ruka-v2-hand.md
  - ./allegro-hand.md
  - ../queries/grasp-policy-selection.md
sources:
  - ../../sources/repos/en02-op.md
summary: "EN02-OP（Westwood Robotics）：GPL 开源三指 7-DoF 末端执行器，Dynamixel XL/XC-330 + 3D 打印 + EN02-PWR 电源板；DIY 成本可至约 200 USD，用于 THEMIS 人形与通用臂挂载。"
---

# EN02-OP（Westwood 开源三指末端）

## 一句话定义

**EN02-OP** 是 [Westwood Robotics](https://www.westwoodrobotics.io/) 发布的 **开源三指末端执行器**：**7 DoF**、**Robotis Dynamixel** 舵机、**3D 打印** 结构体与单块 **EN02-PWR** 电源/通信板；源码与制造文件在 [Westwood-Robotics/EN02-OP](https://github.com/Westwood-Robotics/EN02-OP)，官方亦将其作为自研全尺寸人形 **THEMIS** 的末端（与 Caltech **Themis** 为不同平台）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DoF | Degrees of Freedom | 本夹爪为 7 个独立可控关节 |
| EE | End-Effector | 机械臂或人形腕部末端执行器 |
| BOM | Bill of Materials | 机械与 PCBA 物料清单 |
| GPL | GNU General Public License | 本仓库默认 v3 开源许可 |
| TTL | Transistor-Transistor Logic | Dynamixel 总线通信电平（非 RS-485） |
| PCBA | Printed Circuit Board Assembly | EN02-PWR 电源/菊花链板 |

## 为什么重要

- **成本与门槛**：相对 [Allegro Hand](./allegro-hand.md)、[RUKA-v2 Hand](./ruka-v2-hand.md) 等高 DoF 仿人手，EN02-OP 用 **三指 7 DoF** 把材料与驱动压到 **约 200 USD** 量级（全 XL-330 + 自装电源板，以官方 README 为准），适合 **入门抓取、遥操作与臂端集成** 实验。
- **制造链闭环**：STEP 总成、分件打印包、机械 BOM、装配 PDF、Gerber/BOM/原理图齐全；唯一强依赖是 **按目标法兰修改掌座** 后再打印。
- **电源适配机器人母线**：EN02-PWR 接受 **20–36 V** 输入、输出 **5 V/10A** 供舵机，便于接到现成移动底座或人形 DC 母线；BOM 另给 **12 V** 改板表以换用其他 XC330 型号。
- **控制栈极简**：无专用策略仓库；直接用 [Robotis Dynamixel SDK](https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/) 做位置/力矩级关节控制，适合与自研 IL/RL 或 MoveIt 臂端规划拼接。

## 核心信息

| 维度 | 规格（公开 README / 仓库） |
|------|---------------------------|
| 构型 | 三指；**7 DoF**；强调多姿态适应不同物体外形 |
| 执行器 | **XC-330-M288-T**（5 V，约 $99/个）或 **XL-330**（轻载降本，约 $26/个） |
| 结构 | 多数件 **3D 打印**；近端等受力件建议 **CF/玻纤增强** 材料 |
| 安装 | 掌座底面 **圆截面定位面**；螺孔需用户按法兰 **改 STEP 后打印** |
| 电子 | **EN02-PWR** 单板：降压 + TTL Dynamixel 菊花链；可购预装约 $35 |
| 许可 | **GPL-3.0**；商用需另洽 Westwood |
| 机构 | Westwood Robotics |

## 制造与集成流程

```mermaid
flowchart LR
  A[改 EN02-PALM_BASE<br/>匹配法兰孔位] --> B[3D 打印<br/>General + CF 件]
  B --> C[订购 Dynamixel<br/>XL/XC-330 ×7]
  C --> D[制板/购 EN02-PWR<br/>5V 或 12V BOM]
  D --> E[按 Assembly Guide<br/>机械装配]
  E --> F[Dynamixel SDK<br/>关节控制 / 臂端规划]
```

## 与相近硬件对照

| 平台 | DoF / 形态 | 成本量级 | 控制栈 | 适用 |
|------|------------|----------|--------|------|
| **EN02-OP** | 三指 **7 DoF** | **~$200** DIY（官方估算） | Dynamixel SDK | 低成本臂端/人形非高灵巧抓取 |
| [RUKA-v2 Hand](./ruka-v2-hand.md) | 仿人 **16+2 DoF** | ~$1.5K 材料 | OpenTeach + BAKU 等全栈 | 高 DoF 遥操作与 IL 研究 |
| [Orca Hand](./orca-hand.md) | 仿生灵巧手 | 官网 BOM | orcahand 生态 | 仿生五指复刻 |
| [Allegro Hand](./allegro-hand.md) | 四指 **16 DoF** | ~$16K 级 | 研究社区驱动 | 标准灵巧操作 benchmark |

## 常见误区

- **掌座不能直接打印**：默认 STEP **无安装孔**；跳过改模会导致无法可靠固定到臂法兰。
- **电压与舵机型号必须一致**：12 V 改板后只能配对应 12 V XC330；上电前应用万用表 **实测 EN02-PWR 输出**，避免 5 V 舵机误接 12 V。
- **THEMIS 同名歧义**：Westwood 全尺寸人形亦称 THEMIS；文献中的 Caltech/JHU **Themis**（如 [MPC-RL](./paper-mpc-rl-humanoid-locomotion-manipulation.md)）是另一台机器。

## 关联页面

- [Manipulation](../tasks/manipulation.md) — 操作任务与末端硬件选型语境
- [抓取专题汇总](../overview/topic-grasp.md) — 感知–规划–执行闭环
- [Orca Hand](./orca-hand.md) / [RUKA-v2 Hand](./ruka-v2-hand.md) — 更高 DoF 开源手对照
- [Query：抓取策略选型](../queries/grasp-policy-selection.md)

## 推荐继续阅读

- [EN02-OP 仓库 README](https://github.com/Westwood-Robotics/EN02-OP) — BOM、电气与装配主文档
- [Build EN02-OP with Kevin（YouTube）](https://www.youtube.com/watch?v=WY0bp_8Il1o) — 逐步组装教程
- [Robotis XC330 e-Manual](https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/) — 通信与控制 API

## 参考来源

- [en02-op.md](../../sources/repos/en02-op.md)
