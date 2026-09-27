# Physical AI is enabled by mechanical hardware（Science Robotics 2026）

> 来源归档（ingest）

- **标题：** Physical AI is enabled by mechanical hardware
- **类型：** paper / perspective / humanoid / hardware / physical-ai / industry-outlook
- **期刊：** *Science Robotics* **11**(117)，2026-08-12
- **DOI：** [10.1126/scirobotics.aee2921](https://doi.org/10.1126/scirobotics.aee2921)
- **PubMed：** [42585280](https://pubmed.ncbi.nlm.nih.gov/42585280/)
- **出版社页：** [science.org/doi/10.1126/scirobotics.aee2921](https://www.science.org/doi/10.1126/scirobotics.aee2921)
- **作者：** Jonathan W. Hurst（通讯 / 唯一作者）
- **机构：** Agility Robotics（Salem, OR, USA）；俄勒冈州立大学（Oregon State University）机械、工业与制造工程学院（Corvallis, OR, USA）
- **作者版 / 非正式全文：** [Agility Robotics — Physical AI Is Enabled By Mechanical Hardware](https://www.agilityrobotics.com/research-analysis/physical-ai-is-enabled-by-mechanical-hardware)（AAAS 许可个人使用；正式版见 DOI）
- **arXiv：** 无
- **代码：** **不适用** — 观点文；无训练/推理仓库
- **入库日期：** 2026-09-27
- **一句话说明：** Hurst 主张 Physical AI 的上限由机械动力学决定：人形应「以人为中心」而非像素级仿人；预测大关节滚动接触摆线 + 高扭矩密度电磁电机、手指腱驱 SEA、足跟冲击与 toe-off 等硬件路径，并反驳「硬件commodity、软件通才」叙事。

## 摘要（Crossref / 期刊）

> We are entering an era where AI enables multipurpose applications, but the mechanical hardware must be done right.

## 开源核查（步骤 2.5，截至 2026-09-27）

| 资源 | 状态 |
|------|------|
| **Science Robotics 正式 PDF** | 出版社订阅 / 机构访问；无 arXiv 作者版 |
| **Agility 作者版网页** | **可读全文**（非 redistribution）；含 Competing Interests（作者持股 Agility） |
| **Digit 控制 / 训练代码** | **未列公开 URL** — 商业产品栈 |
| **论文配套代码 / 数据** | **不适用** — Perspective，非系统论文 |

**结论：** **不适用（无可运行官方论文代码）** — 价值在硬件选型与产业判断，非复现栈。

## 核心摘录（面向 wiki 编译）

### 1) AI 受硬件物理「囚禁」

- **要点：** 工业臂擅长高精度轨迹；人类擅长复杂地形、移动操作与力交互。仅软件 / AI 不能弥合两类能力鸿沟；**关节摩擦、惯量、柔顺性** 才是控制（含 AI）作用世界的唯一通道。
- **对 wiki 的映射：** [paper-physical-ai-mechanical-hardware](../../wiki/entities/paper-physical-ai-mechanical-hardware.md)；[humanoid-actuator-102-compliance-sensing](../../wiki/overview/humanoid-actuator-102-compliance-sensing.md)

### 2) 「硬件 commodity」叙事过火

- **要点：** 人形进入公共视野时，有预测认为硬件将商品化、机器几乎完全由软件定义。作者认为 **过犹不及** — 外形可像人，**动力学不必**；早期仿鸟扑翼类比：理解 loco-manipulation 物理之前，不必追求五指手与人等尺寸。
- **对 wiki 的映射：** 同上；[notable-commercial-robot-platforms](../../wiki/overview/notable-commercial-robot-platforms.md)

### 3) 人形形态的第一性原理

- **要点：** 双足 +  upright torso + 双臂 + 交互用「脸」；**窄 footprint**（走廊）、**腿比轮快响应扰动**、躯干容纳算力与电池、肩位双臂最大化地面—高处的操作包络与平衡惯量辅助。
- **对 wiki 的映射：** [locomotion](../../wiki/tasks/locomotion.md)

### 4) 执行器与传动预测

- **大关节：** 电磁电机（轴向 / 横向磁通等）提高 **扭矩密度**；**滚动接触 cycloid** 传动 — 低摩擦、耐冲击、紧凑、高带宽力控与 **反驱**。
- **手：** **腱驱 + SEA**；执行器在 forearm；高减速、非直接反驱但需 **力控与冲击耐受**；腱为 **可更换磨损件**（类比鞋底）。
- **足：** 功能 **heel strike** 吸收触地冲击；**toe-off** 在近乎直膝 gait 下注入步态能量；结构超越「平板金属脚」。
- **对 wiki 的映射：** [humanoid-actuator-102-decision-species](../../wiki/overview/humanoid-actuator-102-decision-species.md)；[paper-digit-humanoid-locomotion-rl](../../wiki/entities/paper-digit-humanoid-locomotion-rl.md)（Digit SEA 路线实例）

### 5) 部署与安全时间线

- **要点：** 今日 Digit 等已在 **为人设计的空间** 做 dull/dirty/dangerous 物流；能力、价格、监管与 **安全认证级执行器力控** 决定从后仓到零售再到家庭的节奏；多用途人形将像 **路面车辆一样多样**。
- **对 wiki 的映射：** [painode-041-agilityrobotics](../../wiki/entities/painode-041-agilityrobotics.md)

## 对 wiki 的映射

- 新建 [paper-physical-ai-mechanical-hardware](../../wiki/entities/paper-physical-ai-mechanical-hardware.md)
- 交叉：[Goal-Oriented Comms for Physical AI](../../wiki/entities/paper-goal-oriented-comms-physical-ai.md)（通信栈视角互补）
- 交叉：[legged robots advances/challenges review](../../wiki/entities/paper-legged-robots-advances-challenges.md)（同刊腿式五柱综述）
- 站点归档：[agility_physical_ai_mechanical_hardware](../sites/agility_physical_ai_mechanical_hardware.md)

## 当前提炼状态

- [x] sources 归档
- [x] 步骤 2.5 开源核查
- [x] wiki 实体升格
