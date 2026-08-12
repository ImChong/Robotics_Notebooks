# Effective Degree / Polynomial Representations（arXiv:2605.29823）

> 来源归档（ingest）

- **标题：** Quantifying and Optimizing Simplicity via Polynomial Representations
- **类型：** paper / ML-theory / generalization / regularization / simplicity-bias / RL
- **arXiv abs：** <https://arxiv.org/abs/2605.29823>
- **PDF：** <https://arxiv.org/pdf/2605.29823>
- **HTML：** <https://arxiv.org/html/2605.29823>
- **代码：** <https://github.com/xinzaixinzai/Effective-Degree> — 归档见 [`sources/repos/effective-degree.md`](../repos/effective-degree.md)
- **项目页：** 无独立 `*.github.io` / lab 项目页；以 arXiv + GitHub 为入口
- **机构：** 清华大学（Tsinghua University）
- **作者：** Tianren Zhang*、Xiangxin Li*、Minghao Xiao*、Guanyu Chen、Feng Chen（* equal contrib.；中文：章天任、李向欣、肖明昊、陈冠宇、陈峰）
- **发表 / 上传：** 2026-05-28（v1）；2026-06-08（v2）；**ICML 2026**
- **入库日期：** 2026-08-06（初入库）；**复核：** 2026-08-12
- **一句话说明：** 用数据相关插值路径上的正交多项式代理刻画网络函数「有效度数」（ED），作为可量化、可微的简洁性度量与正则，在分类 / CLIP 微调 / Procgen PPO 上提升泛化。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| arXiv | [2605.29823](https://arxiv.org/abs/2605.29823) | 论文与附录（v2） |
| 代码 | [xinzaixinzai/Effective-Degree](https://github.com/xinzaixinzai/Effective-Degree) | 官方实现：相关实验、grokking、ED 正则、RL |
| 对照基线 | SAM / ASAM / Jacobian reg. / Mixup | 文中主对比正则族 |

## 开源状态（步骤 2.5，截至 2026-08-12 复核）

- **已开源：** 官方仓 README 标明 ICML 2026 官方实现；含 `train_wd_regular_torch.py`、`poly/`、`grokking/`、`wise-ft/`、`bert/`、`rl/ppo_procgen.sh` 等可辨识训练/评测入口（2026-08-12 再核：默认分支 tip 仍为 2026-05-11，入口未变）。
- **许可证：** 仓库 API 未返回 SPDX license（截至 2026-08-12）。
- **项目页：** 无独立项目页；代码可用性以 GitHub 为准。
- **处理：** wiki 写「已开源」并保留 `## 源码运行时序图`；互链 `sources/repos/effective-degree.md`。

## 摘要级要点

- **问题：** 深度网有 simplicity bias，但缺少同时满足「跨任务架构通用 / 可大规模量化 / 可微优化」的简洁性度量；sharpness、范数等参数空间代理对重参数化敏感。
- **主张：** 在函数空间用多项式代理：沿数据分布采样插值路径拟合正交多项式，用系数加权的 **Effective Degree (ED)** 度量非线性复杂度。
- **理论：** Thm 3.1 — 对多元多项式，随机插值路径几乎必然保持代数度数序；ED 比裸代数度数对拟合噪声更稳。
- **可微正则：** \(\mathcal{L}=\mathcal{L}_{\mathrm{task}}+\lambda\,\widehat{\mathrm{ED}}\)；分类可用 label-anchored 边界锚定缓解与 CE 冲突。
- **结果要点：**
  - CIFAR-10 ViT-Tiny：Baseline 87.80 → **ED 90.82**（超 SAM/ASAM/Jacobian/Mixup）
  - ImageNet scratch ViT-S/16：original 71.37→**72.76**；strong 74.42→**75.01**
  - CLIP FT：ViT-B/32 ID 76.20→**77.14**，Avg OOD 44.04→**45.31**；B/16 同向
  - GLUE（BERT）：RTE / MRPC / CoLA 均优于 baseline 与 embedding mixup
  - Procgen PPO：Dodgeball / Fruitbot / Jumper / StarPilot 未见 level 泛化提升
- **局限：** 当「更简单特征」易学但非鲁棒目标时 ED 可能失败（附录 H）；训练因额外插值点有可接受开销。

## 核心摘录（面向 wiki 编译）

### 1) 插值路径上的一元限制

\[
\mathbf{x}(\alpha)=\alpha\mathbf{x}_1+(1-\alpha)\mathbf{x}_2,\quad
g(\alpha)=f(\mathbf{x}(\alpha)),\quad
P(\alpha)=\sum_{k=0}^{K}c_k\,T_k(2\alpha-1)
\]

离散输入（文本）在 embedding 空间插值；可选路径内 PCA 压缩高维输出。

### 2) Effective Degree

\[
\mathrm{ED}(P)=\sum_{k=0}^{K}|c_k|\,k,\qquad
\mathrm{ED}_{\mathrm{norm}}(P)=\frac{\sum |c_k|k}{\sum |c_k|}
\]

网络级估计：对多条随机路径取期望 \(\widehat{\mathrm{ED}}(f)\)。

### 3) 可微实现

阻尼最小二乘 \((\mathbf{T}^\top\mathbf{T}+\epsilon I)\mathbf{c}=\mathbf{T}^\top\mathbf{y}\)，经 `LinearSolve` 反传；分类用 label-anchored ED。

### 4) 默认效率向超参（README）

\(r=4\)，\(d_{\max}=3\)，\(n_p=\mathrm{batch}/2\)；再主要调 \(\lambda\)。

## 对 wiki 的映射

- 新建实体页：[wiki/entities/paper-effective-degree.md](../../wiki/entities/paper-effective-degree.md)
- 交叉：[深度学习基础](../../wiki/concepts/deep-learning-foundations.md)、[强化学习](../../wiki/methods/reinforcement-learning.md)、[PPO](../../wiki/methods/ppo.md)、[AdamW](../../wiki/methods/adamw.md)
