# State–Readout Decoupling for Latent World Models（Agentic Intelligence Lab PDF）

> 来源归档（ingest）

- **标题：** State–Readout Decoupling for Latent World Models
- **类型：** paper / latent world model / rollout interface / goal-conditioned planning
- **PDF：** <https://agentic-intelligence-lab.org/files/SRD.pdf>
- **机构主页：** <https://agentic-intelligence-lab.org/>
- **作者：** Xianxin Lai、Ziyi Ding、Weiyu Chen、Xiao-Ping Zhang、Jiayu Chen†（通讯）
- **机构：** 香港大学；清华大学深圳国际研究生院；INFIFORCE Intelligent Technology
- **入库日期：** 2026-09-09
- **一句话说明：** 指出自回归 latent rollout 的 **state–readout coupling** 会把中间读数误差直接喂回后续转移；SRD 用规划视界 hidden state 轨迹承载动力学、latent 仅作读出头，在 LeWM/PLDM 四任务 **8 组设定中 7 组** 提成功率，预测器参数 **−53~58%**、评测时间平均 **−44%**。

## 开源状态（项目页核查，2026-09-09）

- **截至入库日未列官方 GitHub：** Agentic Intelligence Lab 主页与 PDF 未给出代码链接；组织仓 [`Agentic-Intelligence-Lab`](https://github.com/Agentic-Intelligence-Lab) 亦无 SRD 对应实现。按「待发布 / 未开源」记录，便于后续 lint 跟进。

## 核心论文摘录（MVP）

### 1) State–readout coupling 与误差反馈

- **链接：** §1；Eq. (7)–(12)；Fig. 1
- **摘录要点：** AR rollout 中 \(\hat z_{t+k}\) 同时是规划读数与下一步转移输入；一阶展开显示 \(e_{k+1}\approx J_k e_k + \epsilon_k\)，中间 latent 误差经 Jacobian 直接反馈。
- **对 wiki 的映射：** [paper-state-readout-decoupling](../../wiki/entities/paper-state-readout-decoupling.md)

### 2) SRD 接口与 GRU 实现

- **链接：** §3；Eq. (13)–(20)
- **摘录要点：** 从 \(z_t\) 初始化 \(h^{\text{init}}\)；GRU 沿完整动作序列推进 \(h_{0:H-1}\)；共享 readout \(R_\phi(h_k)\to \hat z_{t+k+1}\)，**不**把 \(\hat z\) 回灌转移。损失为视界对齐 MSE \(\mathcal L^{\text{SRD}}\) + 骨干原有 \(\mathcal L_{\text{aux}}\)。
- **对 wiki 的映射：** [paper-state-readout-decoupling](../../wiki/entities/paper-state-readout-decoupling.md)

### 3) LeWM/PLDM 八设定与长视界 TwoRoom

- **链接：** Tab. 1–2；Fig. 2
- **摘录要点：** 四任务 × 两骨干：7/8 成功率提升；LeWM-TwoRoom **88.67%→96.00%**（+7.33 pp）。长视界 Heval 5→20：SRD 成功率 **96%→46%** vs AR **88%→14%**；H=20 时评测时间 **515s→68s**（−86.8%）。
- **对 wiki 的映射：**
  - [paper-lewm](../../wiki/entities/paper-lewm.md)
  - [paper-traj-lewm](../../wiki/entities/paper-traj-lewm.md)
  - [paper-causalvae-world-models](../../wiki/entities/paper-causalvae-world-models.md) — 同作者组

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-state-readout-decoupling.md`](../../wiki/entities/paper-state-readout-decoupling.md)
- 项目页归档：[`sources/sites/agentic-intelligence-lab-srd.md`](../sites/agentic-intelligence-lab-srd.md)
