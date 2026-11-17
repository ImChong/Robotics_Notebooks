---
type: concept
tags: [sea, series-elastic-actuator, resonance, actuator, force-control, damping, notch-filter, disturbance-observer, humanoid]
status: complete
updated: 2026-11-17
summary: "SEA 谐振抑制：串联弹性引入电机与负载的相对振动模态；通过扫频辨识、物理阻尼、带宽/陷波整形与模型观测器，在稳定裕度、力控带宽和柔顺之间取舍。"
original:
  - ../../sources/blogs/wechat_embodied_daily_sea_resonance_suppression_2026_11_11.md
sources:
  - ../../sources/blogs/wechat_embodied_daily_sea_resonance_suppression_2026_11_11.md
related:
  - ./impedance-control.md
  - ./robot-joint-scheme-selection.md
  - ./implicit-explicit-actuator-modeling.md
  - ../methods/joint-actuator-parameter-identification.md
  - ../overview/humanoid-actuator-102-compliance-sensing.md
  - ../../roadmap/depth-torque-motor-design.md
---

# SEA 谐振抑制 / Series Elastic Actuator Resonance Suppression

## 一句话定义

**串联弹性执行器（SEA）的谐振抑制**，是在保留关节被动柔顺和可测弹簧力矩的同时，识别并控制电机–弹簧–负载的相对振动模态，避免其在反馈、冲击或周期性任务中被放大。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SEA | Series Elastic Actuator | 电机传动链与输出负载之间有串联弹性件的执行器 |
| FSEA | Force-Sensing Series Elastic Actuator | Paine 等人区分的一种通过串联弹簧测力的构型 |
| RFSEA | Reaction Force Series Elastic Actuator | Paine 等人比较的另一种反作用力式 SEA 构型 |
| DOB | Disturbance Observer | 利用测量量与标称模型估计未建模扰动的观测器 |
| PD | Proportional–Derivative | 比例–微分反馈；过大反馈增益可能激励弹性模态 |
| FRF | Frequency Response Function | 扫频取得的幅相频率响应，用于识别谐振及稳定性 |

## 为什么重要

- **柔顺不是免费带宽：** SEA 的弹簧可改善低阻抗接触、过载缓冲和力矩测量，却会在传动链中引入可被激发的机械模态。
- **症状不等于原因：** 力矩命令平滑而输出抖动、增益提高后啸叫、着地弹跳与力测量噪声，都需要结合频谱、传感器和控制环节排查，不能一律归因于 PID。
- **选型与部署关联：** 弹簧刚度、等效惯量、摩擦阻尼、采样延迟、滤波器以及接触负载共同决定可安全使用的控制带宽。

## 核心原理

### 先区分单质量近似与两惯量模态

附件给出的单等效惯量模型是

\[
f_0 \approx \frac{1}{2\pi}\sqrt{\frac{k}{J_\mathrm{eq}}},
\]

其中旋转关节的 \`k\` 为扭转刚度（N·m/rad）、\`J_\mathrm{eq}\` 为转动惯量（kg·m²）；原文使用线性质量 \`m\` 记号，两者不能混用单位。

更接近实际旋转 SEA 的简化**双惯量**模型为

\[
\begin{aligned}
J_m\ddot\theta_m+b_m\dot\theta_m+\tau_s &= \tau_m,\\
J_l\ddot\theta_l+b_l\dot\theta_l-\tau_s &= \tau_{\mathrm{ext}},\\
\tau_s&=k(\theta_m-\theta_l)+c(\dot\theta_m-\dot\theta_l).
\end{aligned}
\]

这里电机侧惯量 \`J_m\` 应统一折算到弹簧所在坐标侧；\`J_l\` 为负载侧惯量，\`c\` 是弹性元件的等效相对阻尼。在忽略摩擦、外部约束且两惯量均自由的理想条件下，相对振动模态满足

\[
\omega_r=\sqrt{k\left(\frac1{J_m}+\frac1{J_l}\right)},\qquad f_r=\frac{\omega_r}{2\pi}.
\]

实际谐振、反谐振位置依赖**被测输入/输出传递函数**及环境阻抗；不能仅凭一条 \`f₀\` 公式就锁定陷波中心。

### 弹性模态如何影响反馈

机械谐振可能使特定频段的力矩/速度响应增大，并在闭环交越频率附近带来显著相位滞后。增益、采样延迟、限幅和非线性可能进一步降低稳定裕度。**带宽跨越机械模态不等于必然不稳定**：Paine 等人的 UT-SEA 原始实验通过基于模型的控制和 DOB，抑制约 19.2 Hz 的峰并实现约 60 Hz 的力控带宽；这是**特定样机、特定控制与测量定义**的结果。

### 流程总览

\`\`\`mermaid
flowchart LR
  REF["目标关节力矩 / 运动命令"] --> CTRL["反馈控制与带宽整形"]
  CTRL --> MOTOR["电机 / 减速传动"]
  MOTOR --> SPRING["串联弹性件 k、c"]
  SPRING --> LOAD["输出负载 / 环境接触"]
  SPRING --> SENSE["弹簧形变与力矩估计"]
  SENSE --> FRF["扫频识别模态与相位裕度"]
  FRF --> CTRL
  SENSE --> OBS["模型 / 扰动观测器（可选）"]
  OBS --> CTRL
  DAMP["被动阻尼（可选）"] --> SPRING
\`\`\`

### 三种抑制路径及代价

| 路径 | 作用位置 | 主要价值 | 代价 / 误用风险 |
|---|---|---|---|
| 被动阻尼 | 机械弹簧并联粘滞/材料阻尼 | 真实耗散相对振动能量，断电也起作用 | 发热、滞回、摩擦、能效与力传感误差；不能无条件说「失去柔顺」 |
| 控制器频带整形 | 交越频率、低通、陷波（notch） | 降低对窄带模态的激励或绕开不利频段 | 滤波带来相位和响应代价；陷波不等于物理阻尼，峰值随负载漂移 |
| 基于模型的补偿 | 状态观测器 / DOB / 模型逆与前馈 | 在有足够辨识、测量与延迟裕度时改善跟踪 | 模型误差、噪声、滤波器鲁棒性与饱和风险；不是简单「抵消掉机械谐振」 |

附件的「阻抗整形」混合提到 notch/低通，工程上应进一步区分：**期望闭环阻抗设计**与**信号滤波/带宽约束**不是同一操作。

## 工程实践

1. **隔离电流环和机械问题：** 安全限幅、固定负载工况下检查电流/电机角/输出角/弹簧形变；先排除编码器、力矩估计和时延错误。
2. **做低激励扫频：** 记录命令到弹簧力矩、命令到输出运动的 FRF、幅值峰、相位与幅相随负载变化；不要用仅一个加速度峰来确定所有反馈回路的 \`f_r\`。
3. **建立模型：** 用阶跃/扫频辨识 \`J_m,J_l,k,c\` 及传动损失，明确电机/输出传感位置与反谐振；必要时分别检查 FSEA/RFSEA 的弹簧内力与输出力。
4. **先保证稳健：** 选择保守的交越频率与增益，审查相位裕度、数字滤波延迟、饱和和限位；若使用陷波，确认中心频率、带宽、深度及负载变化后仍有效。
5. **逐级比较：** 基线反馈 → 适量物理阻尼或滤波整形 → 模型前馈/DOB；在相同负载和激励下复测力矩误差、超调、调节时间、谐振峰、峰值电流、温升与着地冲击。
6. **真机安全：** 先台架低能量验证再转机器人接触工况，特别监测弹性件最大形变/应力、机械限位和扭矩饱和。

**一手实验证据：** Paine、Oh、Sentis（2014）的 Fig. 8–10 显示，同一 UT-SEA 平台上提出的控制器较简单 PD 明显减小谐振峰与跟踪误差，并报告约 60 Hz 力控带宽；该结论不能替换实际模组的扫频验收。

## 局限与风险

- **范围：** 本页只讨论单关节弹性模态。多关节耦合、结构共振和柔性足端传递须另做整机辨识。
- **负载改变频谱：** 自由空间与刚性接触、穿戴式与落足工况具有不同边界条件。
- **风险不能绝对化：** 不是所有 SEA 都必然在相同频率出现明显峰，也不是所有跨过谐振的控制都失稳；闭环稳定必须针对完整系统验证。
- **源材料核验边界：** 附件中关于特定商业人形产品方案历史、匿名工程案例和供应链依赖的陈述没有可追溯证据，本页**不采纳为已证实事实**；附件 AI 配图只作解释而非测量。
- **无独立源码：** 这是一条控制概念综合笔记，不对应单篇研究论文或官方可运行仓库，因此不虚构源码运行时序图或开源状态。

## 关联页面

- [阻抗控制](./impedance-control.md) — 对外希望表现的柔顺/刚度与实际弹性模态的区别。
- [机器人关节方案选型](./robot-joint-scheme-selection.md) — SEA 与 QDD、刚性传动力矩测量路径的取舍。
- [隐式与显式执行器建模](./implicit-explicit-actuator-modeling.md) — 在 Sim2Real 模型中保留真实弹性与延迟。
- [关节执行器参数辨识](../methods/joint-actuator-parameter-identification.md) — 用台架数据估计惯量、摩擦和传动误差。
- [Humanoid Actuator 102：柔顺与力感知](../overview/humanoid-actuator-102-compliance-sensing.md) — 传感结构与模组综合约束。
- [力矩控制电机设计纵深路线](../../roadmap/depth-torque-motor-design.md) — Stage 5 的带宽测量和模态诊断。

## 参考来源

- [原文归档：SEA 谐振抑制（第 31 讲）](../../sources/blogs/wechat_embodied_daily_sea_resonance_suppression_2026_11_11.md)（用户提供完整 PDF；文章为工程科普资料，不等同于原始实验论文）
- [Paine, Oh & Sentis, 2014：Design and Control Considerations for High-Performance SEAs](https://sites.utexas.edu/hcrl/wp-content/uploads/sites/3888/2016/01/tmech-paine.pdf)（作者机构全文，DOI: 10.1109/TMECH.2013.2270435，重点核验 Sec. III–IV、Fig. 8–10）
- [Williamson, 1995：Series Elastic Actuators，MIT AITR-1524](https://dspace.mit.edu/entities/publication/7ca5684c-f297-40cc-b145-532eb634fae4)（原始理论与带宽/柔顺取舍，核验摘要）

## 推荐继续阅读

- [Paine 的 High-Performance Series Elastic Actuation 博士论文（UT Austin，2014）](https://sites.utexas.edu/hcrl/wp-content/uploads/sites/3888/2016/01/nick-paine-thesis-dissertation-2014.pdf) — 系统化讨论 SEA 力控、两种传动布置和模型观测器。
- [IHMC：Series Elastic Actuators](https://www.ihmc.us/series-elastic-actuators/) — SEA23-23 的公开 CAD、装配与设计资源（用于理解机构，而非本文代码实现）。
