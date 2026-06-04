# Pupper v3 官方文档（Read the Docs）

> 来源归档

- **标题：** Pupper V3 Documentation
- **类型：** site（官方文档）
- **来源：** Hands-On Robotics / Stanford 课程生态
- **链接：** https://pupper-v3-documentation.readthedocs.io/en/latest/index.html
- **入库日期：** 2026-06-04
- **一句话说明：** Pupper v3 开源伴侣四足的完整建造、运维、安全、技术规格与开发指南；含仿真 RL 训练、VLM 语音交互与 Stanford CS 123 课程入口。

## 文档结构（站点导航）

| 分区 | 页面 | 路径（相对 latest/） |
|------|------|----------------------|
| Make Pupper | 采购清单 | `guide/sourcing_parts.html` |
| | 组装 | `guide/building.html` |
| | 系统安装 | `guide/software_installation.html` |
| Use Pupper | 安全 | `using_pupper/safety.html` |
| | 操作 | `using_pupper/operation.html` |
| | 维护 | `using_pupper/maintenance.html` |
| Take Our Course | Robotics & AI 课程 | `course/course.html` → [Stanford CS 123](https://cs123-stanford.readthedocs.io/en/latest/) |
| Learn More | 技术规格 | `learn_more/tech_specs.html` |
| | 帮助 | `learn_more/help.html` |
| Development | 修改代码 | `development/modifying_code.html` |
| | Foxglove 可视化 | `development/visualization.html` |
| | Pupper AI（进行中） | `development/ai.html` |

## 摘录要点

- **定位：** 面向教学与陪伴的 **开源四足**；自组成本约 **$2000**；强调在人周围（含儿童）使用的 **安全设计**。
- **能力叙事：** 仿真 **强化学习** 越障行走；并行仿真实现「约 1 小时 ≈ 3 个月训练」量级加速叙述；**OpenAI Realtime API** 自然语音；**VLM** 听看与表达（相机、麦克风、本体感知 + 屏幕/耳朵/扬声器）。
- **课程：** 多年 Stanford 授课沉淀；课程站见 CS 123 readthedocs。
- **社区：** [Hands-On Robotics Discord](https://discord.gg/qbmaU8NmP2)；赞助方 [Hands-On Robotics](https://www.handsonrobotics.org)。

## 技术规格（摘自 tech_specs）

| 类别 | 要点 |
|------|------|
| 机构 | **12 DoF**（每腿 3）；蹲姿约 **25×20×22 cm**；质量约 **3 kg** |
| 算力 | **Raspberry Pi 5 8GB**；M.2 可扩展 AI 加速器 |
| 执行器 | **Steadywin GIM4305**（4005 无刷 + 10:1 行星减速）；峰值 ~**3.5 N·m**、风冷连续 ~**1.0 N·m**、最高 ~**30 rad/s**；耳朵 **9g 舵机** |
| 传感 | **BNO086** 6DoF IMU；鼻端 **IMX296** 鱼眼（~222° FOV）；麦克风；驱动器回报关节角/速/力矩；电池电压 ADC |
| 交互 | **4×4 LCD** 表情/调试；鼻后 **3W** 扬声器 |
| CAD | Onshape / Fusion 360 模型链接见文档 |

## 软件与工程

- 机载软件 monorepo：[Nate711/pupperv3-monorepo](https://github.com/Nate711/pupperv3-monorepo/tree/main)（机内路径 `~/pupperv3-monorepo`，ROS 2 工作区 `~/pupperv3-monorepo/ros2_ws`）。
- 远程开发：`ssh pi@pupper.local`；Foxglove Studio 可视化见 `development/visualization.html`。
- **Pupper AI** 页标注 work in progress（麦克风、扬声器、相机自检流程）。

## 安全设计（摘自 safety）

- 无外露夹点/锐边；电池坚固外壳；**力矩限制 + 关节柔顺**；背部 **E-STOP**；小轻量降低风险。
- 使用建议：勿触碰运动部件；勿置于高处；电机可能高温；儿童需监护。

## 采购入口（文档内链，易变）

- BOM 表格（Google Sheets，见 `guide/sourcing_parts.html`）
- 成品套件商链（如 Present Perfection、AIFitLab 等，以文档为准）

## 对 wiki 的映射

- [Stanford Doggo / Pupper（实体页）](../../wiki/entities/stanford-doggo-and-pupper.md) — 补充 **Pupper v3** 硬件、软件栈、RL/VLM 与课程索引
- [pupperv3_monorepo（代码仓归档）](../repos/pupperv3_monorepo.md)
- [easy_quadruped](../repos/easy_quadruped.md) — 仍对应 **早期 Pupper / StanfordQuadruped** 模型控制 lineage，与 v3 栈不同
