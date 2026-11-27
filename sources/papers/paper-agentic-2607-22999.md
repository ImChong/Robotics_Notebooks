# WCM: World-Cognition Model for Generalizable Human-Robot Interaction

- **作者：** Yuzhen Chen, KC Zhou
- **机构：** 哈佛大学（Harvard University）、麻省理工学院（Massachusetts Institute of Technology, MIT）（资助与知识产权归属：Cyberbrain, Inc.）
- **版本 / 载体：** arXiv:2607.22999v2（v1 提交于 2026-07-25，v2 修订于 2026-08-03；arXiv HTML 全文）
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2607.22999) · [arXiv HTML 全文](https://arxiv.org/html/2607.22999)
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，论文全文与 arXiv 元信息未列出 GitHub 仓库或独立项目页链接；致谢声明系统、数据集及相关知识产权归 Cyberbrain, Inc. 所有）
- **核对范围：** 全文通读 Section I（Introduction）、Section II（Method: SLAK Architecture、Asynchronous Runtime、CoT Distillation from Rollouts and Teaching）、Section III（Experiments: Hardware、Nine-Task Suite、Case-Study Comparison、Ablations）、Section IV（Conclusion）、Acknowledgments，以及 Appendix A（Table I, Figs. 3–11 说明）、Appendix B（Table II）、Appendix C（Table III）
- **通读：** 已完成

## 原文核对笔记

1. **核心问题与定位：** 针对现有 VLA（Vision-Language-Action）策略与世界模型规划器偏重开环/单向指令执行、缺乏执行原因可解释性与执行中人类纠偏/教学闭环的问题，提出以人为中心的具身智能体 **World-Cognition Model (WCM)**。
2. **SLAK 四层显式架构：**
   - **Sensing（感知层）：** 融合 RGB-D（$640 \times 480$）、音频、IMU 与力/力矩信号，生成包含局部 3D 几何关系、占据、语义以及像素/部件级交互线索（抓取、开启、放置、丢弃）与粗粒度物理属性（刚性、易碎性、可抓性）的交互感知场景状态 $s_t$。
   - **Logic（逻辑层）：** 基于场景状态 $s_t$、用户指令 $u_t$ 与记忆 $m_t$，运行增量式“规划-推理-验证”闭环，输出 $(r_t, a_t, \hat{o}_t) = f_\theta(x_t)$（推理轨迹、动作、预期结果），并在执行前后按 $v_t = \mathbb{I}[\Delta(o_t, \hat{o}_t) \le \epsilon]$ 校验实际结果 $o_t$ 与预期结果 $\hat{o}_t$ 的偏差；若未通过则重新推理或向用户澄清。
   - **Knowledge（知识层）：** 维护短期任务状态（当前目标、观察对象、空间上下文、近期动作/结果/用户反馈）与长期经验（任务经验、物体交互历史、技能记忆、失败与教学记录）。
   - **Action（动作层）：** 将逻辑层选定步骤绑定至跨任务复用的底层技能库（导航、伸手、抓取、放置、交接、简单关节物体交互），向 Logic 和 Knowledge 回传状态与失败信息。
3. **异步运行时（Asynchronous Runtime）：** 推理、硬件执行与状态更新并发解耦；在提交队列动作前及执行后持续应用 Eq. (1) 校验，上下文变化时丢弃过期队列动作并重规划，支持边移动、边解释、边响应人类纠偏。
4. **基于 CoT 蒸馏的自举数据飞轮与人机教学：**
   - 将自主执行（含成功与失败）$\mathcal{D}_{\mathrm{auto}}$ 与自然语言人机教学片段 $\mathcal{D}_{\mathrm{teach}}$ 统一记录为 $d_t = (s_t, u_t, m_t, r_t, a_t, \hat{o}_t, o_t)$。
   - 通过更强模型集成（voting/consensus）经离线精炼 $\mathcal{R}(\cdot)$ 去除冗余绕路并修正失败推理轨迹，得到 $\mathcal{D}'_{\mathrm{auto}}$ 与 $\mathcal{D}'_{\mathrm{teach}}$，再以加权负对数似然 $\mathcal{L}_{\mathrm{distill}}(\theta) = \mathcal{L}_{\mathrm{auto}}(\theta) + \lambda \mathcal{L}_{\mathrm{teach}}(\theta)$（$\lambda \ge 1$）蒸馏到推理模型，监督推理过程而非底层电机动作。
5. **硬件与实机评测（Table I–III）：**
   - 平台为低成本移动操作机器人（麦克纳姆轮底盘 + 5-DOF 约 1.3 m 臂展机械臂 + $640 \times 480$ RGB-D 相机，本体成本低于 $2,000；推理运行在局域网外接共享 NVIDIA RTX 5090）。
   - 9 项真实世界 HRI 任务平均成功率 **73.8%**（含 4 项从 CoT 微调中保留的 held-out 任务）；长时序任务“Take the screwdriver to the drawer.”在教学前成功率低于 20%，交互教学后达 **69%**（9/13），CoT 蒸馏后升至 **82%**。
   - 与 X-Square WALL-OSS 的 6 任务案例对比中，WCM 平均移动耗时 38.5 s（对照 103.5 s）、平均 GPU 占用时间 22.0 s（对照估计 103.5 s），6 任务平均成功率 88.0%（注：两者机器人本体、运动学、传感器与执行栈不同，论文明确声明该对比为说明性案例而非受控基准）。
   - 5 任务消融实验表明：去掉异步运行时使端到端耗时增至 **1.4–1.7×**；将交互感知层替换为普通 VLM + 2D grounding 后，部件级交互任务成功率大幅下跌（其中 Screwdriver-to-Drawer 从 **69%** 降至 **0%**）。

对 wiki 的映射：[WCM: World-Cognition Model for Generalizable Human-Robot Interaction](../../wiki/entities/paper-agentic-2607-22999.md)
