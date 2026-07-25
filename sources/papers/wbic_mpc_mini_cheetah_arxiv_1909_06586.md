# Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control

> 来源归档

- **标题：** Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control
- **类型：** paper
- **作者：** Donghyun Kim, Jared Di Carlo, Benjamin Katz, Gerardo Bledt, Sangbae Kim
- **机构：** MIT
- **链接：** https://arxiv.org/abs/1909.06586
- **PDF：** https://arxiv.org/pdf/1909.06586
- **代码：** https://github.com/mit-biomimetics/Cheetah-Software
- **入库日期：** 2026-07-25
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
