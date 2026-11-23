---
type: source-paper
updated: 2026-11-22
arxiv: "2108.06266"
url: https://arxiv.org/abs/2108.06266
---

# Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning (Annual Review 2022)

- **作者：** Lukas Brunke, Melissa Greeff, Adam W. Hall, Zhaocong Yuan, Siqi Zhou, Jacopo Panerati, Angela P. Schoellig
- **arXiv：** <https://arxiv.org/abs/2108.06266>（Annual Review of Control, Robotics, and Autonomous Systems, 2022）
- **通读：** 已完成
- **核对范围：** 已通读 arXiv:2108.06266v2 全文 PDF（3,677 行），重点核对控制论与 RL 三分法统一框架（不确定动力学学习 / 鼓励安全的 RL 与 CMDP / 可认证安全过滤器 predictive-HJ-CBF backup 策略，§ 4 Benchmarks）。
- **关键核对要点：**
  - 统一比较三大谱系：安全动力学学习（GP/MPC 鲁棒集）、安全 RL（CMDP / 风险敏感 / 域随机化）、以及基于预测/CBF/Hamilton-Jacobi 可达集的 **可认证安全过滤器（Safety Filter）**。
  - § 4 实证统计揭示许多早期 Safe RL 仅在确定性仿真评测且开源率偏低，真机部署亟需不确定性包络与备份策略验证（u_certified 保持 Ω_safe ⊂ X_c）。
