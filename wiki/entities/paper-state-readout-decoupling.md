---
type: entity
tags: [paper, world-models, latent-dynamics, planning, rollout, gru, tsinghua, hku]
status: complete
updated: 2026-09-09
venue: "预印本 2026（Agentic Intelligence Lab PDF；无 arXiv 编号、无会议）"
related:
  - ./paper-lewm.md
  - ./paper-traj-lewm.md
  - ./paper-causalvae-world-models.md
  - ../methods/generative-world-models.md
  - ../concepts/latent-imagination.md
sources:
  - ../../sources/papers/srd_state_readout_decoupling.md
  - ../../sources/sites/agentic-intelligence-lab-srd.md
summary: "SRD（Agentic Intelligence Lab PDF）：用规划视界 GRU hidden state 承载 rollout、latent 仅作读出头，消除 state–readout coupling；LeWM/PLDM 四任务 8 设定中 7 组提成功率，预测器参数 −53~58%、评测时间平均 −44%。截至入库日未开源。"
---

# SRD：latent 世界模型的状态–读数解耦

**State–Readout Decoupling (SRD)**（[PDF](https://agentic-intelligence-lab.org/files/SRD.pdf)，香港大学 Agentic Intelligence Lab）针对 reconstruction-free latent WM 的 **自回归 rollout**：每步预测 latent \(\hat z_{t+k}\) 既给规划器打分，又作为下一步转移输入（**state–readout coupling**），使中间读数误差经 predictor Jacobian **直接反馈**到后续步。SRD 改为沿完整动作序列推进 **GRU hidden state**，latent 预测仅经共享 readout 暴露给规划器，**不回灌**转移。

## 一句话定义

**把多步 latent rollout 的原生对象从「递归拼接的 latent 链」改成「规划视界上的 hidden state 轨迹」，用视界对齐 MSE 监督每一步 readout，从而切断 latent 读数误差对后续动力学的直接反馈路径。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SRD | State–Readout Decoupling | 本文 rollout 接口重构 |
| AR | Autoregressive Latent Rollout | 基线：一步预测器递归调用 |
| LeWM | LeWorldModel | 实验骨干之一 |
| PLDM | Planning with Latent Dynamics Models | 实验骨干之二 |
| CEM | Cross-Entropy Method | 目标条件 latent 规划器（未改） |
| GRU | Gated Recurrent Unit | SRD 匹配实现中的状态载体 |

## 为什么重要

- **机制清晰：** 给出 \(e_{k+1}\approx J_k e_k+\epsilon_k\) 的误差反馈公式，把 compounding error 归因到接口设计而非单纯「模型太小」。
- **工程双赢：** 在 LeWM/PLDM 上 **7/8** 设定提成功率，同时预测器参数 **−53~58%**、评测时间平均 **−44%**（长视界 H=20 时 **−86.8%**）。
- **与 LeWM 生态衔接：** 编码器、目标表征、CEM、重规划循环 **不变**——只换 rollout 模块，便于 ablation。
- **同组拼图：** [CausalVAE](./paper-causalvae-world-models.md) 管因果因子；[Traj-LeWM](./paper-traj-lewm.md) 管轨迹打分；SRD 管 **rollout 载体**。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 香港大学；清华大学深圳国际研究生院；INFIFORCE Intelligent Technology |
| **任务** | Two-Room、Push-T、OGB-Cube、Reacher（视觉目标到达） |
| **骨干** | LeWM、PLDM（仅换 predictor + 训练目标） |
| **训练视界** | \(H_{\text{train}}=5\)；长视界诊断 \(H_{\text{eval}}\in\{5,10,15,20\}\) |
| **开源** | **未开源**（截至 2026-09-09：PDF / Lab 页 / GitHub 均无官方实现链接） |

## 流程总览

```mermaid
flowchart TB
  subgraph AR [基线 AR]
    Z0[z_t] --> P1[Predictor]
    P1 --> Z1["ẑ_{t+1} → 输入下一步"]
    Z1 --> P2[Predictor]
    P2 --> Z2["ẑ_{t+2}"]
  end
  subgraph SRD [SRD]
    ZT2[z_t] --> INIT[g_φ 初始化 h]
    INIT --> GRU[GRU 沿 a_{t:t+H-1}]
    GRU --> HSEQ["h_0…h_{H-1}"]
    HSEQ --> READ[R_φ readout]
    READ --> ZSEQ["ẑ_{t+1: t+H}（仅读数）"]
  end
```

## 核心原理

### State–readout coupling

AR：\(\hat z_{t+k+1}=P_\phi(\hat z_{t+k},a_{t+k})\)。同一 \(\hat z_{t+k}\) 承担 **监督/目标距离** 与 **下一转移状态** 两角色。

### SRD 接口

\[
h_0=F_\phi(h^{\text{init}},u_0),\quad h_k=F_\phi(h_{k-1},u_k),\quad \hat z_{t+k+1}=R_\phi(h_k)
\]

**\( \hat z \) 不出现在 \(F_\phi\) 的参数中。** 损失：

\[
\mathcal L^{\text{SRD}}=\frac{1}{H}\sum_{k=1}^{H}\mathrm{MSE}(\hat z_{t+k},z^{\text{tgt}}_{t+k}),\quad \mathcal L=\mathcal L^{\text{SRD}}+\mathcal L_{\text{aux}}
\]

\(\mathcal L_{\text{aux}}\) 为 LeWM/PLDM 原有表征损失（不变）。

### SRD-AR 变体

可选加一步局部转移辅助头 \(\mathcal L^{\text{AR}}\)（训练期 only）；**评测仍用解耦 rollout**。

## 源码运行时序图

截至入库日 **无官方可运行仓库**：

| 项 | 说明 |
|----|------|
| 源码运行时序图 | **不适用** — Agentic Intelligence Lab 主页与 [SRD.pdf](https://agentic-intelligence-lab.org/files/SRD.pdf) 未列出 GitHub；组织仓无对应项目。 |
| 复现路径 | 在 LeWM/PLDM 上替换 AR predictor 为「GRU 状态推进 + 共享 readout + 视界 MSE」；规划栈保持 CEM 不变。 |

## 实验与评测

### 八设定成功率（相对 AR）

| 骨干 | 最大增益示例 | 唯一退步 |
|------|-------------|----------|
| LeWM | Two-Room **+7.33 pp**（88.67→96.00%） | Push-T **−7.34 pp**（接触几何敏感） |
| PLDM | Push-T SRD-AR **+11.33 pp** | — |

- **效率：** 预测器参数 **−53.0~58.2%**；评测时间 **−24.2~62.8%**（均值约 **−44%**）。
- **长视界 TwoRoom（LeWM）：** H=20 成功率 SRD **46%** vs AR **14%**；开环终端 MSE **−17.2%**；H=20 评测 **515s→68s**。
- **消融：** 仅加多步监督（AR-MS）几乎无效；**换载体为 hidden state**（AR-GRU 或完整 SRD）才带来主要增益。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| **AR 自回归 rollout**（本文要替代的基线） | 本页最硬的一组：编码器、目标表征、CEM、重规划循环全部不变，只换 predictor 与训练目标。8 设定里 7 组提成功率，预测器参数 −53~58%、评测时间均值 −44%（H=20 时 −86.8%）。唯一退步是 LeWM 上的 Push-T（−7.34 pp），说明接触精细任务不吃这套 |
| **AR-MS（只加多步监督）/ AR-GRU（只换载体）** | 消融给出的因果归属：只加多步监督几乎无效，**换载体为 hidden state** 才是增益来源。读法上不要把 SRD 记成「多步 loss」，它是接口重构 |
| **SRD-Transformer 变体** | 载体换成 Transformer 可拿到部分增益但不及 GRU 版；GRU 是实现选择而非机制本身，但「随便换个序列模型都行」也不成立 |
| [CausalVAE WM Plug-in](./paper-causalvae-world-models.md) | 同组、同一 WM 栈的另一处改动，切口不同：CausalVAE 改 latent 因子的**因果可识别性**，SRD 改**谁承载多步动力学**。论文互引但模块独立，未联合实验 |
| [Traj-LeWM](./paper-traj-lewm.md) / [LeWM](./paper-lewm.md) | LeWM 是本文的实验骨干之一；Traj-LeWM 改的是**规划打分**（轨迹代价），SRD 改的是**rollout 载体**，二者正交，论文明确提可组合但未做 |
| [生成式世界模型](../methods/generative-world-models.md) | 范式边界：本文全程在 reconstruction-free latent WM 里讨论，视频生成式 WM 的 rollout 不存在同一个 state–readout coupling，结论不可直接搬 |
| [潜空间想象](../concepts/latent-imagination.md) | Dreamer 系同样用循环状态推进想象，与 SRD 的差别在于**latent 预测是否回灌转移**——这正是本文归因 compounding error 的那条路径 |

## 结论

**总判：SRD 用接口层改动同时改善长视界成功率、开环误差与 rollout 算力——证明 latent WM 规划应先问「谁在承载多步动力学」，而不只是加深一步预测器。**

1. 长视界 / 导航类任务优先试 SRD 式解耦；接触精细任务（LeWM-PushT）可能退步，需任务级验证。
2. 与 LeWM 正交扩展：[Traj-LeWM](./paper-traj-lewm.md) 改打分，SRD 改 rollout——可组合实验。
3. GRU 是实现选择，核心是 **解耦**；SRD-Transformer 可达部分增益但不如完整 SRD+GRU。
4. **未开源** — 复现需按论文式(13)–(20) 自实现并对接现有 LeWM/PLDM 编码器与 CEM。
5. 论文引用 [CausalVAE](./paper-causalvae-world-models.md) 为同组因果动力学线，但模块独立。

## 局限与风险

- Push-T 上 LeWM 退步说明 horizon-level state 不总是优于逐步 latent 接触建模。
- 无官方代码，复现与超参对齐成本高。
- 仅评估目标条件 CEM 四任务，未覆盖视频生成 WM 或真机 VLA。

## 关联页面

- [LeWM](./paper-lewm.md)
- [Traj-LeWM](./paper-traj-lewm.md)
- [CausalVAE WM Plug-in](./paper-causalvae-world-models.md)
- [潜空间想象](../concepts/latent-imagination.md)
- [生成式世界模型](../methods/generative-world-models.md)

## 参考来源

- [SRD 论文归档](../../sources/papers/srd_state_readout_decoupling.md)
- [Agentic Intelligence Lab 归档](../../sources/sites/agentic-intelligence-lab-srd.md)
- PDF：<https://agentic-intelligence-lab.org/files/SRD.pdf>

## 推荐继续阅读

- [LeWM](./paper-lewm.md)（[arXiv:2603.19312](https://arxiv.org/abs/2603.19312)）
- [Traj-LeWM](./paper-traj-lewm.md)（同组路径感知扩展）
