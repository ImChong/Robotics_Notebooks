# Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control

> 来源归档

- **标题：** Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control
- **类型：** paper
- **作者：** Donghyun Kim, Jared Di Carlo, Benjamin Katz, Gerardo Bledt, Sangbae Kim
- **机构：** MIT
- **链接：** https://arxiv.org/abs/1909.06586
- **PDF：** https://arxiv.org/pdf/1909.06586
- **代码：** https://github.com/mit-biomimetics/Cheetah-Software
- **入库日期：** 2026-07-25（首次）；2026-11-07（全文通读更新）
- **版本 / 日期：** arXiv v1，2019-09-14
- **核对范围：** 通读摘要、§I 引言、§II 混合架构、§III MPC（凸 MPC 的三条简化假设、QP 代价、摩擦锥线性化、步态调度与落脚点规划）、§IV WBIC（零空间优先级逆运动学、QP、力矩解算、Table I 任务与增益）、§V 实验（3.7 m/s 跑步、室外多步态）与 §VI 结论；参考文献仅浏览。论文**未报告** MPC 预测时域长度与 Q/R 数值（引用其前作 Di Carlo 2018，该文本次未取到）。
- **通读：** 已完成
- **一句话说明：** Mini Cheetah 经典分层栈：长时域简化模型 **MPC** 求反力剖面 + **WBIC（Whole-Body Impulse Control）** 求关节力矩/位置/速度，面向腾空相与高速摆腿。
- **开源状态：** **已开源**（Cheetah-Software 含 MPC/WBIC 实现入口）
- **沉淀到 wiki：** [paper-wbic-mpc-mini-cheetah](../../wiki/entities/paper-wbic-mpc-mini-cheetah.md)

---

## 核心贡献（摘录）

1. MPC 用简化模型在较长时域优化地面反力；WBIC 基于该反力计算关节指令。
2. 相对「只跟踪躯干轨迹」的 WBC，本框架更强调冲量/反力一致性，利于动态步态与空中相。
3. 在 Mini Cheetah 上验证高度动态 locomotion。

## 对 wiki 的映射

- [paper-wbic-mpc-mini-cheetah](../../wiki/entities/paper-wbic-mpc-mini-cheetah.md)
- [mpc-wbc-integration](../../wiki/concepts/mpc-wbc-integration.md)
- [srbd-convex-mpc-wbc](../../wiki/concepts/srbd-convex-mpc-wbc.md)


## 核心摘录补充（2026-11-07，均出自原文）

- **架构与频率：** 凸 MPC（简化质心 / 刚体模型，约 30–40 Hz，图中标 0.04 kHz）求反力剖面；WBIC 以 500 Hz 运行；关节层 PD 为 40 kHz；机载低功耗计算机即可，作者自研动力学引擎（含转子动力学）。
- **WBIC 与普通 WBC 的区别：** 优先目标是**跟踪 MPC 的反力指令**而不是躯干轨迹；对浮动基座加速度加松弛变量 $\delta_f$，允许基座偏离指令（腾空相 CoM 不可控）。作者明确选择"松弛浮动基座动力学"而不是用 HQP 严格层级。
- **优先级逆运动学：** 用零空间投影 $J_{i|pre}=J_iN_{i-1}$、$N_0=I-J_c^\dagger J_c$ 递推出 $\Delta q,\dot q^{cmd},\ddot q^{cmd}$（式 16–20），加速度部分用动力学一致伪逆；输出关节位置 / 速度指令给关节 PD。
- **QP（式 25）：** $\min \delta f_r^TQ_1\delta f_r+\delta f^TQ_2\delta f$，s.t. 浮动基座动力学、加速度指令、$f_r=f_r^{MPC}+\delta f_r$、接触约束 $Wf_r\ge0$；权重 $Q_1=1$、$Q_2=0.1$。最后由式 26 解出关节力矩。QP 求解器为 Goldfarb–Idnani 对偶法（文献 [21]，适合小规模）。
- **增益（Table I）：** 身体姿态 / 位置 / 足端任务 $k_p=100\,s^{-2}$、$k_d=10\,s^{-1}$；关节 PD $k_p=3$ N·m/rad、$k_d=0.3$ N·m·s/rad（外展关节 $k_d=1$）；所有测试共用同一组参数与任务设置。
- **结果：** Mini Cheetah 跑速 3.7 m/s（Froude 数 4.65；观测峰值 4 m/s 后失稳），髋关节峰值速度 34 rad/s、力矩 25.5 N·m（命令值，超出额定 17 N·m 时执行器截断），说明用满硬件能力；同一参数跑六种步态（含 gallop、pronk、bound）。
