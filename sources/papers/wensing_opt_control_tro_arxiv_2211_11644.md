---
type: source-paper
updated: 2026-11-21
arxiv: "2211.11644"
url: https://arxiv.org/abs/2211.11644
---

# Optimization-Based Control for Dynamic Legged Robots (T-RO 2024 Survey)

- **作者：** Patrick M. Wensing, Michael Posa, Yue Hu, Adrien Escande, Nicolas Mansard, Andrea Del Prete
- **arXiv：** <https://arxiv.org/abs/2211.11644>（IEEE Transactions on Robotics, 2024）
- **通读：** 已完成
- **核对范围：** 已通读 arXiv:2211.11644v1 全文 PDF（2,715 行），重点核对 § III–V（SRBD/质心动力学 MPC 与 NMPC）、§ VI（QP-based WBC vs HQP 词汇表优先序、CLF-QP/CBF 融合）与 § VII（Contact-Implicit TO、DDP vs 直接转录、Model-Based 与 RL 融合展望）。
- **关键核对事实：**
  - 经典 MPC+WBC 分层：长时域简化模型预测地面反力（如 MIT Convex MPC / Centroidal NMPC）下发给瞬时全阶动力学 QP-WBC 求解 `min w1||J_G ν̇ + J̇_G ν - a_des||^2 + w2||τ||^2`，受约束于浮动基动力学 `Mν̇ + Cν + τ_g = S^T τ + J_c^T λ`、接触加速度 `J_c ν̇ = a_c` 与摩擦锥 `C λ <= 0`。
  - **加权 QP vs HQP：** 单级 QP 实现简单、兼容主流开源求解器（OSQP/qpOASES 等），但权重难调且无法严格隔离多优先级不等式；HQP 为极限权重比的严格超集，可借助专用单遍求解器加速，但受奇异性与实现复杂度限制。
  - **NMPC 与求解趋势：** 相比固定接触时序 MPC，Contact-Implicit TO 用互补或平滑柔性接触摆脱预设步态；DDP/iLQR 适合高自由度序列化推进，直接配点/多重打靶则更利于并行化。
