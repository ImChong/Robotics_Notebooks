# 动量的新理解：逼近特征层面的梯度下降（科学空间 #11875）

- **类型**：blog
- **作者**：苏剑林（BoJone）
- **原始链接**：<https://kexue.fm/archives/11875>
- **发布日期**：2026-08-23
- **收录日期**：2026-08-24
- **站点索引**：[kexue-fm-scientific-spaces.md](../sites/kexue-fm-scientific-spaces.md)

## 一句话

把 **动量** 重新诠释为「让参数更新 $W\leftarrow W-\eta\Phi$ 所诱导的特征变化 $X\Phi$ 逼近 **特征层梯度下降** $\partial L/\partial Y$」的 **在线线性回归解**；由此自然导出 **Newton-Muon**（显式输入预条件 + `msign`）与 **DeltaMomentum**（内层梯度更新、免矩阵求逆）。

## 为什么值得保留

- **统一 Muon 变体动机**：将 SGDM、Newton-Muon、Muon 放在同一「特征梯度 ↔ 参数更新」框架下；各向同性时 Newton-Muon 退化为 Muon（与 [11549](https://kexue.fm/archives/11549) 结论一致）。
- **动量机制新视角**：动量不仅是梯度 EMA，还可视为最小化 $\|\|X\Phi - \partial L/\partial Y\|\|_F^2 + \lambda\|\|\Phi\|\|_F^2$ 的解；与 **GDN / MesaNet**、线性注意力 **Vanilla → DeltaNet → MesaNet** 演变类比。
- **工程权衡清晰**：指出输入预条件需在前向保存 $X^\top X$、破坏优化器黑盒独立性，以及子空间探索不足等理论与实现代价。

## 核心论点（原文归纳）

### 1. 优化器骨架

动量状态 $M_t=\beta M_{t-1}+(1-\beta)G_t$，更新 $W_t=\phi(W_{t-1},M_t,G_t,t)$；差异在 $\phi$（SGDM、SignSGD、Muon 等）。本文改的是对 **动量含义** 的理解，而非仅改 $\phi$。

### 2. 线性层与两种梯度下降

设 $Y=XW$，$G=\partial L/\partial W = X^\top \partial L/\partial Y$。

- **参数层**：$W\leftarrow W-\eta G$
- **特征层（理想）**：$Y\leftarrow Y-\eta \partial L/\partial Y$ — 但 $Y$ 不可直接改，只能通过 $W$ 间接实现

### 3. 回归目标 → 预条件动量

令 $W\leftarrow W-\eta\Phi$，则 $Y\leftarrow Y-\eta X\Phi$。希望 $X\Phi\approx \partial L/\partial Y$，解：

$$
\min_\Phi \frac{1}{2}\|X\Phi - \partial L/\partial Y\|_F^2 + \frac{\lambda}{2}\|\Phi\|_F^2
\quad\Rightarrow\quad
\Phi^*=(X^\top X+\lambda I)^{-1}G
$$

对 $X^\top X$ 与 $G$ 做 EMA 得 **SGDM 变体**：$Z_t=\beta Z_{t-1}+(1-\beta)(X_t^\top X_t+\lambda I)$，$W\leftarrow W-\eta Z_t^{-1}M_t$。

### 4. Newton-Muon

将 $Z_t^{-1}M_t$ 视为「更靠谱的动量」，代入 Muon 的 `msign`：

$W\leftarrow W-\eta\,\mathrm{msign}(Z_t^{-1}M_t)$ — 大致对应 [Newton-Muon](https://arxiv.org/abs/2604.01472)（原文可先校正再 EMA，省一组状态）。输入各向同性时 $Z_t\approx \sigma^2 I$，退化为 Muon；$\lambda\to\infty$ 亦退化 Muon。

### 5. DeltaMomentum

不对 $\Phi$ 求闭式解，对内层目标用梯度下降迭代 $\Phi$，得 [DeltaMomentum](https://arxiv.org/abs/2608.19491) — 类比 GDN 相对 Vanilla Linear Attention；$Z^{-1}M$ 类比 MesaNet。实现需处理 $X$ 归一化与内层学习率 $\gamma$；**无需矩阵求逆**。

### 6. 相关工作与局限

- 谢天知乎文：谱范数约束下特征最速下降，$Z^{-1/2}$ 内外作用于 `msign`。
- Newton-Muon 在 Speedrun 上优于 Muon（[官方实现](https://github.com/zhehangdu/Newton-Muon)）。
- **局限**：需耦合前向保存 $X^\top X$；预条件可能限制在非输入张成子空间；$\lambda$ 为额外超参。

## 对 wiki 的映射

- [wiki/concepts/feature-space-gradient-descent.md](../../wiki/concepts/feature-space-gradient-descent.md) — 新建概念页
- [wiki/methods/muon.md](../../wiki/methods/muon.md) — 补充特征层视角与 Newton-Muon / DeltaMomentum
- [wiki/methods/sgd-momentum.md](../../wiki/methods/sgd-momentum.md) — 交叉「回归解」解读
- [sources/papers/muon_optimizer_primary_refs.md](../papers/muon_optimizer_primary_refs.md) — 索引 DeltaMomentum

## 参考链接

- 原文：<https://kexue.fm/archives/11875>
- Newton-Muon：<https://arxiv.org/abs/2604.01472> · 代码 <https://github.com/zhehangdu/Newton-Muon>
- DeltaMomentum：<https://arxiv.org/abs/2608.19491>
- 各向同性背景：<https://kexue.fm/archives/11549>
- 矩阵 r 次方根：<https://kexue.fm/archives/11175>
