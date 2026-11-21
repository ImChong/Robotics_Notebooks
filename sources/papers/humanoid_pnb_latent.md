# LATENT: Learning Athletic Humanoid Tennis Skills from Imperfect Human Motion Data（arXiv:2603.12686）

> 来源归档（ingest · 原文通读）

- **标题：** LATENT: Learning Athletic Humanoid Tennis Skills from Imperfect Human Motion Data
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/LATENT__Learning_Athletic_Humanoid_Tennis_Skills_from_Imperfect_Human_Motion_Dat/LATENT__Learning_Athletic_Humanoid_Tennis_Skills_from_Imperfect_Human_Motion_Dat.html>
- **分类：** 04_Loco-Manipulation_and_WBC
- **arXiv：** <https://arxiv.org/abs/2603.12686>（v1，2026-03-13）
- **PDF：** <https://arxiv.org/pdf/2603.12686>
- **项目页：** <https://zzk273.github.io/LATENT/>
- **代码：** <https://github.com/GalaxyGeneralRobotics/LATENT>
- **机构：** Tsinghua University；Peking University；Galbot Inc.；Shanghai Qi Zhi Institute；Shanghai AI Laboratory
- **入库日期：** 2026-06-07
- **核查日期：** 2026-11-21
- **一句话说明：** LATENT 只用 5 小时、3 × 5 米小场地采集的"业余网球动作碎片"，就把 Unitree G1 训练成会在真人对打下完成连续多拍回合的"人形网球手"——核心办法是先用动作跟踪器学出一个可修正的 latent 动作空间，再让高层策略在该空间里做 "修正 + 组合"，并用 Latent Action Barrier (LAB) 约束策略别跑出先验分布。
- **核对范围：** 通读 arXiv:2603.12686v1 全文正文（§1 Introduction、§2 Related Work、§3 LATENT 三阶段方法与式 (1)–(4) / Table 1 / Figs. 2–4、§4 Sim-to-Real Transfer 与 Table 2、§5 Experiments 与 Tables 3–5、§6 Conclusion）及附录 Appendix A–B（运动覆盖与仿真自博弈 Figs. 5–8）。
- **通读：** 已完成

## 核心技术与实验事实（原文核对）

- **不完美人动作数据（§3.1）：** 5 名业余网球手在 $3\,\text{m}\times 5\,\text{m}$ 光学动捕区内采集 **5 小时**基本动作碎片（正手、反手、侧移步、交叉步等），无编辑与标注，通过 **LocoMuJoCo** 重定向到 29-DoF Unitree G1；数据同时存在**不精确**（手腕挥拍细节失真）与**不完整**（缺少打网球决策组合知识）。
- **可修正隐动作空间（§3.2）：**
  1. 基于 Any2Track 框架在 MuJoCo JAX（50 Hz 控制、2000 Hz 物理仿真，8 GPU PPO）预训练低层跟踪器，训练时**剔除持拍右手腕控制信号并注入随机手腕扰动**，迫使全身对腕部修正保持平衡鲁棒。
  2. 通过 DAgger 在线蒸馏为条件变分信息瓶颈（VIB）：编码器 $E(z_t^q|s_t,\tilde{s}_{t+1})$、解码器 $D(\hat{a}_t^{body}|s_t,z_t^q)$ 与**可学习条件先验** $P(z_t^p|s_t)=\mathcal{N}(\mu_p(s_t),\sigma_p(s_t))$（替代固定单峰高斯先验），保持右手腕无控与扰动训练。
- **高层策略、动作修正与 LAB（§3.3）：**
  - $\pi_{planner}(s_t, g_t, b_t)$ 输出混合动作 $[a_t^{latent}, a_t^{correct}]$：前者经 **Latent Action Barrier（LAB）** 按状态相关标准差 $\sigma_t^p$ 构造马氏距离界限，将最终全身 PD 目标约束为 $a_t^{full}=[D(s_t, \mu_t^p + \lambda \sigma_t^p \cdot \tanh(a_t^{latent})), a_t^{correct}]$，既避免策略恶意利用隐空间产生高频抖动（policy hacking），又由 $a_t^{correct}$ 直接补偿右手腕精确挥拍角度。
- **Sim2Real（§4）：** 随机化机器人足底/关节摩擦、电枢惯量、质心/质量、球拍质心/质量及网球质量、弹性恢复、切向阻尼和空气阻力（$f_{air}=-k m v \|v\|$）；对球位速度采用 **4 帧滑动平均**抑噪并注入观测噪声/丢帧/延迟。真机 Unitree G1 拆下右手并通过 3D 打印件固定网球拍，由 $19\,\text{m}\times 15\,\text{m}$ 区域内 50+ 台 120 Hz 动捕相机追踪机器人 6D 位姿与贴反光膜的网球。
- **关键实验（§5 & 附录）：**
  - **Table 3（仿真 10,000 trials，来球峰值速度 >15 m/s，2.5 m 落点阈值）：** 正手 SR **96.52%**（DE 1.32 m）、反手 **82.10%**（1.89 m）、前场 **86.35%**（1.73 m）、后场 **89.80%**（1.55 m），显著高于 PULSE（71.85% / 63.38% / 74.52% / 72.94%）、ASE 与 AMP（PPO 与 MotionVAE 均不收敛）。
  - **Table 4（消融）：** 去掉右手腕修正（w/o Corr.）反手 SR 跌至 68.94%；去掉 LAB（w/o LAB）平滑度从 25.61 恶化至 37.64、力矩从 7.40 升至 12.53 且各指标全面下滑。
  - **Table 5（真机 20 次人机对打回合，有效落入对方半场）：** 正手 **90.90%**、反手 **77.78%**、前场 **88.89%**、后场 **81.82%**（去掉球动力学随机化后暴跌至 14.29%–28.57%）。
  - **Appendix A.2（仿真自博弈）：** 两台策略对打在 50 局随机测试中最高连续 **25 拍**回合。

## 对 wiki 的映射

- [paper-notebook-latent](../../wiki/entities/paper-notebook-latent.md)
- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
