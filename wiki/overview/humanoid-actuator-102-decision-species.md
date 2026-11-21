---
type: overview
tags: [humanoid, actuator, tesla-optimus, unitree, agility-digit, category-hub]
status: complete
updated: 2026-11-21
summary: "Actuator 102 · 07 — No Free Lunch 评分卡；三大物种（工厂/快递/家庭）；关节指标 >15Nm/kg、反向驱动 <1Nm、峰值持续 3:1、智能关节集成。"
related:
  - ../concepts/robot-joint-scheme-selection.md
  - ./humanoid-actuator-102-technology-map.md
  - ./humanoid-actuator-102-compliance-sensing.md
  - ./humanoid-actuator-102-industrial-actuator-trap.md
  - ../queries/humanoid-hardware-selection.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_actuator_102.md
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
original:
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
---

# Actuator 102 · 07：决策矩阵与三大物种

> **分类页**：**X 总决策矩阵** + **XI 关节执行器设计要求**。

## 一句话定义

「没有免费午餐」执行器决策矩阵按照人形机器人的目标应用场景将主流路线分化为工业重资产搬运（滚柱直线+谐波）、高动态敏捷巡检（SEA/QDD）与静音家庭伴侣（低比谐波或 QDD）三大物种，并给出 70 kg 级下肢主关节的核心量产设计基线（比力矩 `>15 Nm/kg`、反驱 `<1 Nm`、峰持比 `>=3:1` 等）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SEA | Series Elastic Actuator | 串联弹性执行器，在负载链串联弹簧吸收冲击并测力 |
| QDD | Quasi-Direct Drive | 准直驱，低减速比与低反射惯量的高频力控方案 |
| G1 | Unitree G1 Humanoid | 宇树低成本通用科研/商业双足人形平台 |
| RL | Reinforcement Learning | 依托高保真物理动力学完成运动策略迁移的核心方法 |

## 为什么重要

执行器研发很容易陷入“参数内卷”（盲目追求极致比力矩、最高减速比或最高电流密度），却在落地产线上发现声音太吵进不了家庭、反射惯量太大摔两次腿就碎、或者刚性模型在仿真训练时彻底跑偏。把所有物理学权衡收敛到“应用场景驱动的物种矩阵与验收硬指标清单”，是整机立项前冻结硬件规格书（PRD/Spec）的最后总闸。

## 核心原理与规格矩阵

### 三大物种分化（原文总决策矩阵）

| 物种 | 核心任务诉求 | 主流执行器架构倾向 | 代表商业/科研案例 |
|------|--------------|--------------------|-------------------|
| **A 工厂工人** | 持续搬重箱、8h+ 班次、精准高负载放置 | 滚柱直线（膝踝）+ 谐波旋转（肩腕髋） | Tesla Optimus、Figure 02、Apptronik Apollo |
| **B 敏捷快递/科研** | 快速机动、耐摔冲击、动态跨越与高步频续航 | SEA 或低比行星 QDD | Agility Digit、Unitree G1 / H1（路线梳理见 [Physical AI × 机械硬件](../entities/paper-physical-ai-mechanical-hardware.md)） |
| **C 家庭助手** | 低噪静音、人机接触安全、极致成本可控 | 低减速高平滑谐波或大转矩无背隙 QDD | 1X Neo（部分线缆/柔驱）、Ubtech Walker |

短期内人形硬件**不会**趋同为单一「通用最佳架构」：高减速高出力 vs 反射惯量、物理弹簧储能 vs 环路带宽等**底层物理约束无法用算法白嫖消除**。

### 仿真友好度与 Sim2Real 维度（第 X 章）

- **刚性机械链（精密谐波、低隙滚柱丝杠）**：物理响应高度接近理想刚体铰链模型，更容易让基于大规模并行物理引擎（Isaac Lab / MuJoCo）的 RL 策略稳定迁移（作者指出这是工厂类人形偏好刚性驱控的深层隐性收益）。
- **弹性/高柔顺系统（SEA、流体液压、强粘滞线缆）**：弹性迟滞、温漂与高维接触间隙会大幅抬高系统辨识及领域随机化难度。

### 下肢主关节核心设计硬指标（第 XI 章摘要）

| 工程维度 | 原文推荐门槛 | 校验注意事项 |
|----------|--------------|--------------|
| 比力矩（Torque/Mass） | 峰值 **> 15 Nm/kg**（一线竞争力 **> 25 Nm/kg**） | 分清含驱动板与减速器后的模组全重 vs 裸定子重量 |
| 反向驱动力矩 | 输出端外力反驱 **< 1 Nm** | 决定地面接触感与冲击瞬时退让能力 |
| 峰值 / 持续转矩比 | **≥ 3:1** | 对照持续温升边界而不仅是瞬态电流上限 |
| 动态响应与带宽 | 力矩环 **50–100 Hz**；指令延迟 **< 1 ms** | 受总线、编码器延时及传动柔性共同限制 |
| 输出端反射惯量 | **< 0.1 kg·m²**（70 kg 级下肢语境） | 受 $N^2 J_m$ 直接支配 |
| 冲击疲劳寿命 | 线接触冲击构型；**> 1000 万次** 循环量级 | 避免使用点接触工业滚珠丝杠硬扛足跟撞击 |

> 膝关节样本规格详见原文图 31；髋、踝、肩、肘、腕按各自载荷谱与摆角范围等比缩放。核对上述比力矩与峰持比数据时，务必在供应商完整的 [TN 曲线](../concepts/motor-torque-speed-curve.md) 与 [TI 曲线](../concepts/motor-torque-current-curve.md) 上逐点核验，防止被单点宣传极限误导。

## 工程实践

- **冻结架构前先定义产品边界**：若目标工位只需要搬运 15–20 kg 工件在平整工厂水泥地面行走，无需为了“好看的后空翻”强行上 QDD 牺牲蹲持热稳定；反之，做跑跳动态运动实验平台切勿直接上全谐波或自锁丝杠。
- **把“声学噪声”与“失电安全掉电表现”写入家庭物种规范**：低减速齿轮噪声远低于高速小行星直驱啸叫，线驱或低阻尼 QDD 也必须考虑断电下蹲缓降机制。
- **基于关节工作空间反向校准膝/踝规格书**：将实际步行峰值着地速度、期望深蹲角度以及总重预算逐项代入 [机器人关节方案选型](../concepts/robot-joint-scheme-selection.md) 工具流程校核。

## 常见误区或局限

- **误区：把单轴峰值力矩参数表直接相加就能评估整机表现** — 忽视持续工作热平衡、直流母线总功率上限以及反驱摩擦力盲区，会导致样机在静态搬重时表现不错、上动态步态一小时后全轴降额保护。
- **时效与数据口径局限**：文中参数门槛基于 70 kg 标准成年体格人形的经验归纳（如 15 Nm/kg 与 1000 万次循环量级），微型双足（如 25–35 kg 级）或超重特种人形需按动力学量纲重新标定。

## 关联页面

- [Actuator 102 技术地图](./humanoid-actuator-102-technology-map.md)
- [机器人关节方案选型](../concepts/robot-joint-scheme-selection.md)
- [电机转矩-转速曲线（TN 曲线）](../concepts/motor-torque-speed-curve.md)
- [电机转矩-电流曲线（TI 曲线）](../concepts/motor-torque-current-curve.md)
- [人形硬件选型 Query](../queries/humanoid-hardware-selection.md)
- [旋转-直线分离架构](./humanoid-actuator-102-split-architecture.md)
- [柔顺与感知反馈](./humanoid-actuator-102-compliance-sensing.md)
- [工业执行器陷阱](./humanoid-actuator-102-industrial-actuator-trap.md)
- [产业与成本](./humanoid-hardware-101-supply-chain-economics.md)

## 推荐继续阅读

- [Humanoid Hardware 101 · 04：集成执行器](./humanoid-hardware-101-integrated-actuators.md) — 单机 BOM 角度的整机关节配比
- [自研关节模组开发流程](../concepts/joint-module-self-development-workflow.md) — 规格落地后的自研五件套与测试闭环

## 参考来源

- [wechat_human_five_humanoid_actuator_102.md](../../sources/blogs/wechat_human_five_humanoid_actuator_102.md)
- [wechat_humanoid_actuator_102_2026-06-02.md](../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md)（第十、十一章）
