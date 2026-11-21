---
type: source-paper
updated: 2026-11-22
arxiv: "1903.11199"
url: https://arxiv.org/abs/1903.11199
---

# Control Barrier Functions: Theory and Applications (Ames et al., ECC 2019)

- **作者：** Aaron D. Ames, Samuel Coogan, Magnus Egerstedt, Gennaro Notomista, Koushil Sreenath, Paulo Tabuada
- **arXiv：** <https://arxiv.org/abs/1903.11199>（ECC 2019 Invited Survey）
- **通读：** 已完成
- **核对范围：** 已通读 arXiv:1903.11199v1 全文 PDF（1,555 行），重点核对 § II CLF/CBF 集合前向不变性（safety vs liveness 对偶）、CLF-CBF-QP 联合公式与松弛变量 δ 折中，以及机器人步态/多机避碰应用（VI. Conclusions）。
- **关键核对要点：**
  - **安全性 vs 活性（Safety vs Liveness）**：CLF 刻画渐近稳定性（liveness，希望状态趋近平衡点），CBF 通过水平集 `C={x: h(x)>=0}` 的前向不变性刻画安全性（safety，“坏状态不发生”）。
  - **CLF-CBF-QP 统一综合**：在线实时 QP 以 `||u - u_nom||^2 + p δ^2` 为代价，将硬安全约束 `L_f h(x) + L_g h(x) u >= -α(h(x))` 与可松弛收敛约束 `L_f V(x) + L_g V(x) u <= -c V(x) + δ` 并行求解，冲突时优先保安全。
