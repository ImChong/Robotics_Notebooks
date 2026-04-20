---
type: task
tags: [manipulation, bimanual, humanoid, dual-arm, whole-body, imitation-learning, teleoperation]
status: stub
summary: "双臂协调操作（Bimanual Manipulation）要求两只手臂在力学和时序上协同完成单臂无法完成的任务，是人形机器人操作能力的核心挑战之一。"
sources:
  - ../../sources/papers/imitation_learning.md
related:
  - ./manipulation.md
  - ./loco-manipulation.md
  - ../concepts/whole-body-control.md
  - ../concepts/whole-body-coordination.md
  - ./teleoperation.md
---

# Bimanual Manipulation（双臂协调操作）

**双臂协调操作（Bimanual Manipulation）**：同时使用两只手臂协同完成一个任务，两臂之间存在物理或时序上的依赖关系——典型任务包括双手递接、拧瓶盖、折叠衣服、组装零件。单臂独立控制无法完成这类任务。

## 一句话定义

双臂操作不是"两个单臂任务的叠加"，而是两臂之间存在**刚性耦合（共同操持物体）**或**协调约束（时序配合）**的整体任务，需要将两臂建模为一个统一系统来规划和控制。

## 为什么重要

人类日常操作任务中，约 70% 需要双手协作：

- 开瓶盖（一手固定，一手旋转）
- 折叠衣物（两手协同对折）
- 端盘子（双手平衡托举）
- 组装 IKEA 家具（一手固定，一手拧螺丝）
- 打包装箱（双手压住箱盖，同时封胶带）

人形机器人的设计初衷就是替代人类完成这类任务，因此双臂操作能力直接决定了人形机器人在实际场景中的可用性。

目前单臂操作（pick-and-place）已相对成熟（ACT、Diffusion Policy 等方法成功率较高），但双臂操作仍是开放挑战：数据采集难、约束建模复杂、策略泛化差。

## 关键挑战

### 1. 双臂闭链约束（Closed-Loop Kinematics）

当两臂同时操持同一物体时，两个末端执行器和物体之间形成**闭合运动链**：

$${}^A T_{obj} \cdot {}^{obj} T_B = {}^A T_B$$

其中 ${}^A T_{obj}$ 是左臂末端到物体的变换，${}^{obj} T_B$ 是物体到右臂末端的变换。这个等式约束两臂不能独立运动——任何一臂的移动都必须由另一臂补偿，否则物体被拉扯变形（或损坏）。

处理这个约束的方式：
- **主从控制（Leader-Follower）**：一臂（主）设定轨迹，另一臂（从）用约束计算随动轨迹
- **混合力/位置控制（Hybrid Force-Position）**：一臂位置控制主导运动，另一臂力控制维持抓持力
- **虚拟物体坐标（Virtual Object Frame）**：以物体坐标系为参考，统一规划两臂在物体空间中的运动

### 2. 内力（Internal Force）管理

即使两臂末端位置满足约束，两臂也可能对物体施加相互抵消的"内力"（internal wrench）。内力不产生净运动，但会造成物体变形、抓持不稳，甚至损坏被操作物。

理想情况：内力最小化（通过优化抓持力分配）
实际情况：内力的测量和控制需要精确的力/力矩传感器

### 3. 协调时序与同步

许多双臂任务要求精确的时序协调，而非同时动作：
- 折叠毛巾：一手翻折，另一手固定，交替进行
- 拧瓶盖：一手的旋转速度必须与另一手的抓紧力同步

学习这类时序依赖关系需要能够捕捉**跨臂时间相关性**的模型架构（如双头 Transformer 或对称 diffusion policy）。

### 4. 数据采集的固有难度

双臂遥操作（bimanual teleoperation）是目前主要的数据来源，但：
- **双手 VR 手套/外骨骼**：设备昂贵，标定复杂，延迟和人因误差累积
- **ALOHA 式双手领导者机器人**：成本低但操作精度受限，依赖操作员技能
- **单机器人遥操作双臂**：认知负载高，人难以自然协调双手
- **MoCap 数据迁移**：运动重定向（motion retargeting）到机器人双臂仍是研究课题

### 5. 策略架构的设计挑战

双臂操作的策略需要处理：
- **高维动作空间**：双臂动作空间维度是单臂的两倍（通常 12~28 维）
- **跨臂依赖**：两臂动作不独立，需要建模它们之间的相互依赖
- **多模态动作分布**：同一任务可以有多种双手配合方式（无唯一最优）

## 常见方法路线

### 路线 A：模仿学习（IL）

**ACT（Action Chunked Transformers）**（Zhao et al., 2023）是目前双臂操作的主流基线：
- 双臂关节状态 + 视觉输入 → Transformer → 动作序列（action chunk）
- 使用 Conditional VAE（CVAE）处理多模态演示数据
- 在 ALOHA 双臂平台上验证，成功率显著优于单步预测方法

**Diffusion Policy（双臂版）**：
- 双臂动作联合建模为扩散过程，自然捕捉多模态分布
- 适合对称或周期性双臂任务

### 路线 B：WBC + 任务空间规划

适合需要精确力控制的工业任务：
- 上层：规划双臂在任务空间的协调运动轨迹（考虑闭链约束）
- 中层：TSID/WBC 将任务空间轨迹转化为关节力矩，同时处理内力优化
- 下层：力/阻抗控制维持接触

**优势**：动力学一致，可精确控制内力
**劣势**：需要精确建模，对未知物体泛化差

### 路线 C：端到端 RL

从仿真中训练双臂 RL 策略：
- 奖励设计极为困难（如何量化"协调"？）
- 真实物体接触建模在仿真中不准确，sim2real gap 严重
- 目前主要在玩具任务（如双臂接球、双臂开门）上有结果

### 近期代表工作

| 工作 | 方法 | 平台 | 关键贡献 |
|------|------|------|---------|
| ALOHA / ACT（2023） | IL + CVAE | ALOHA 双臂 | 定义了双臂 IL 的基线范式 |
| Mobile ALOHA（2024） | IL + 移动底盘 | ALOHA + 底盘 | 双臂 + 移动的融合 |
| π₀（2024） | VLA + Flow Matching | 人形/桌面 | 统一 loco-manip 的双臂策略 |
| GELLO（2023） | 低成本遥操作 | 各类机械臂 | 降低双臂数据采集门槛 |

## 与 Loco-Manipulation 的区别

| 维度 | 双臂操作（Bimanual） | 移动操作（Loco-Manip） |
|------|-------------------|--------------------|
| 底盘 | 固定或移动底盘 | 必须移动（行走） |
| 核心挑战 | 双臂协调约束 / 内力 | 全身协调 / 行走稳定性 |
| 数据采集 | 双臂遥操作 | 全身动作捕获 |
| 控制复杂度 | 双臂耦合 | 腿+臂耦合（更高） |
| 当前成熟度 | 有较多研究基线 | 仍是前沿开放问题 |

双臂操作是 loco-manipulation 的子集：一旦机器人开始行走，双臂操作的所有挑战都会叠加上行走稳定性的额外复杂度。

## 评价指标

- **任务成功率**：完成整个双臂任务的比率（而非单臂子步骤）
- **操作精度**：末端位置误差、物体损坏率
- **协调性**：两臂时序一致性、内力大小
- **泛化能力**：对新物体、新摆放位置的适应率
- **数据效率**：完成任务所需演示数量

## 关联系统/方法

- [Manipulation](./manipulation.md) — 双臂操作的理论基础
- [Loco-Manipulation](./loco-manipulation.md) — 双臂操作 + 移动的扩展
- [Whole-Body Control](../concepts/whole-body-control.md) — 处理双臂闭链约束的控制框架
- [Whole-Body Coordination](../concepts/whole-body-coordination.md) — 多肢体协调的概念框架
- [Teleoperation](./teleoperation.md) — 双臂数据采集的主要手段
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md) — 双臂操作大量涉及接触丰富型场景（组装、插拔、折叠）
- [Action Chunking](../methods/action-chunking.md) — 双臂协调常依赖动作块输出保持短时同步

## 参考来源

- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (RSS, 2023) — ALOHA + ACT 奠基工作
- Fu et al., *Mobile ALOHA* (2024) — 移动双臂操作
- [sources/papers/imitation_learning.md](../../sources/papers/imitation_learning.md) — ACT / ALOHA / Diffusion Policy 摘要

## 推荐继续阅读

- Zhao et al., [*ACT: Action Chunked Transformers*](https://arxiv.org/abs/2304.13705)
- [ALOHA](https://tonyzhaozh.github.io/aloha/) — 低成本双臂遥操作平台
- [Whole-Body Coordination](../concepts/whole-body-coordination.md)
- [Query：接触丰富操作实践指南](../queries/contact-rich-manipulation-guide.md) — 双臂装配与内力协调的实践排障思路

## 一句话记忆

> 双臂操作的核心不是"两个手各做各的"，而是建模两臂之间的物理耦合和时序协调约束，用统一框架让两只手真正"合作"而非"各自为政"。
