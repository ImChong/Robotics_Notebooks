# HumanoidArena: Benchmarking Egocentric Hierarchical Whole-body Learning（arXiv:2606.17833）

> 来源归档（ingest）

- **标题：** HumanoidArena: Benchmarking Egocentric Hierarchical Whole-body Learning
- **缩写：** **HumanoidArena**
- **类型：** paper / benchmark / humanoid / hierarchical-control / egocentric / HOI / HSI
- **arXiv：** <https://arxiv.org/abs/2606.17833>（PDF：<https://arxiv.org/pdf/2606.17833>）
- **项目页：** <https://humanoidarena.github.io>
- **发表日期：** 2026-06-16
- **作者：** Taowen Wang, Zikang Xie, Bin Yang, Yunheng Wang, Zizhao Yuan, Yuetong Fang, Yixiao Feng, Yichi Wang, Xingyu Chen, Haodong Chen, Qiwei Wu, Weisheng Xu, Lihan Chen, Lusong Li, Zecui Zeng, Renjing Xu（* 共同一作以项目页为准）
- **机构：** 香港科技大学（广州）、北京工业大学、哈尔滨工业大学（深圳）、深圳北理莫斯科大学、京东探索研究院
- **入库日期：** 2026-07-06
- **一句话说明：** **仿真优先** 的 egocentric **分层全身学习基准**：高层策略把第一人称视觉、本体与指令映射为 **40D 中间全身动作**，由低层 **GMT**（TWIST2 / SONIC）执行；在 **7 项下肢关键 HOI/HSI** 任务上，从 **扰动泛化** 与 **跨 GMT 迁移** 两轴诊断 policy–tracker 接口——分层控制能解多样腿关键交互，但性能 **强 tracker 条件化**、**跨 GMT 迁移脆弱**。

## 摘录 1：问题与分层 formulation（摘要 / 引言）

- **痛点：** 人形全身交互需要 **任务级决策** 与 **全身动态执行** 紧耦合；实用路线是 **分层控制**——高层预测 **中间全身动作**，低层 **General Motion Tracker (GMT)** 稳定执行。但既有基准很少 **直接评测 policy–tracker 接口**：中间动作是否可执行、在任务分布偏移下是否鲁棒、能否 **跨不同 GMT 后端迁移**。
- **核心问法：** 在共享 **egocentric 分层接口** 下，模仿学习与 VLA 风格高层策略学到的 **中间全身表征** 是否 **可转移**、对 **视觉/语义/执行扰动** 与 **GMT 换后端** 是否稳健？
- **与相邻路线：** 相对 TeleOpBench / HumanoidMimicGen 等 **采集或合成数据基准**，HumanoidArena 强调 **腿足结构必要** 的 HOI/HSI 与 **双 GMT 后端** 下的 **接口诊断**；相对纯 tracking 基准，它评测 **高层策略 + 低层 tracker 闭环**。

**对 wiki 的映射：**
- [HumanoidArena](../../wiki/entities/paper-humanoidarena.md) — 问题定义、任务套件与评测轴
- [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 分层全身 loco-manip 基准语境
- [Whole-Body Control](../../wiki/concepts/whole-body-control.md) — 高层–低层分工

## 摘录 2：七项下肢关键任务与评测协议（§Task / Evaluation）

- **任务套件（7）：** 强调 **下肢协调结构必要**，而非把腿当平面运输工具：
  - **HOI：** Football（踢球与平衡）、DoubleDesk（跨台面搬运与全身转向）、P&PBox（蹲姿高位放置）
  - **HSI：** OpenDoor（把手操作与穿门）、SitSofa（障碍导航与坐姿过渡）、Boxing（蹲击与高度自适应）、VisNavi（受限场景 egocentric 视觉导航）
- **双 GMT 后端：** **TWIST2** 与 **SONIC** 在 Isaac Lab 内以 **不同低层跟踪动力学** 解释同一上游 **35D 共享机器人空间参考**。
- **评测轴（四类扰动 + 跨 GMT）：**
  - **Base：** 默认设置基线
  - **Semantic：** 高层任务/环境语义变化（如球门外观替换）
  - **Vision：** 观测视觉偏移（如光照方向）
  - **Execution：** 控制侧退化（如足球初始化范围扩大 → 不同接近几何与触球时机）
  - **Cross-GMT：** 固定高层策略、**换低层 GMT 后端**，测中间动作表征可迁移性
- **Baselines：** 代表性 **模仿学习** 与 **VLA 风格** 高层策略，在共享接口下对比。

**对 wiki 的映射：**
- [HumanoidArena](../../wiki/entities/paper-humanoidarena.md) — 任务表与 Football 扰动示例
- [TWIST2](../../wiki/entities/paper-twist2.md) — GMT 后端之一
- [SONIC](../../wiki/methods/sonic-motion-tracking.md) — GMT 后端之二

## 摘录 3：采集–规范化–训练管线（§Pipeline）

- **1 共享采集与重定向：** 操作员经 **PICO 头显** 接收人形 **egocentric 视频**；人体动作经 **GMR** 重定向为 **35D 机器人空间参考信号**，供实时消费。
- **2 后端特定动作解释：** Isaac Lab 内 **TWIST2** 或 **SONIC** action provider 将共享信号解释为可执行 **G1 目标**，暴露不同低层跟踪动力学。
- **3 录制与规范化：** Recording manager 序列化 **NPZ**（egocentric 观测、状态、动作、可回放轨迹）；示范规范为 **64D canonical state** 与 **40D intermediate whole-body action**（含主 ego、左右腕视角）。
- **4 转换与 benchmark：** 原始录制转 **LeRobot 兼容数据集** → 训练高层策略 → 在 **in-GMT / cross-GMT / visual / semantic / execution** 协议下评测。
- **开源资源：** 代码、LeRobot 数据集、策略 checkpoint、Isaac Lab 仿真资产（项目页 GitHub / HuggingFace / Google Drive 入口）。

**对 wiki 的映射：**
- [HumanoidArena](../../wiki/entities/paper-humanoidarena.md) — Mermaid 管线总览
- [Teleoperation](../../wiki/tasks/teleoperation.md) — PICO egocentric 遥操作采集栈
- [Motion Retargeting (GMR)](../../wiki/methods/motion-retargeting-gmr.md) — 共享上游参考语义

## 摘录 4：主要实验结论（§Results）

- **分层控制有效性：** 在共享 policy–tracker 接口下，所学高层策略可解决 **多样腿关键交互**（项目页展示 P&PBox、Football、OpenDoor 等成功/失败/超时对比）。
- **Tracker 条件化：** 同一高层策略在不同 GMT 后端下性能差异显著——**性能强 tracker 条件化**。
- **跨 GMT 迁移脆弱：** **Cross-GMT** 协议显示：固定高层、换 TWIST2↔SONIC 时，**中间全身动作表征迁移仍脆弱**——定位 HumanoidArena 为研究 **可转移中间动作表征** 与 **可扩展 egocentric 全身策略学习** 的基准。
- **扰动敏感性：** Football 等任务上 **Vision / Semantic / Execution** 轴使 **外观、语义与执行几何** 敏感性可分解诊断（项目页以 Football 为具体示例）。

**对 wiki 的映射：**
- [HumanoidArena](../../wiki/entities/paper-humanoidarena.md) — 结论与误区
- [Imitation Learning](../../wiki/methods/imitation-learning.md) — IL 高层策略 baseline 语境
- [VLA](../../wiki/methods/vla.md) — VLA 风格高层策略对照
