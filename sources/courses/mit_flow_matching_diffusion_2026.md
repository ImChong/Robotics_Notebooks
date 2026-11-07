# MIT 6.S184：Flow Matching and Diffusion Models（2026）

> 来源归档（ingest · MIT CSAIL IAP 2026）

- **标题：** Introduction to Flow Matching and Diffusion Models
- **类型：** course
- **课号：** MIT 6.S184 — *Generative AI with Stochastic Differential Equations*（IAP 2026；亦列 6.S975）
- **站点：** <https://diffusion.csail.mit.edu/2026/index.html>
- **讲义 notes：** 站点 *Course Notes* 按钮（自包含 PDF；cite 见下）
- **Lab 仓库：** <https://github.com/eje24/iap-diffusion-labs/tree/2026>（MIT License）
- **讲义源码仓：** <https://github.com/eje24/iap-diffusion-class/>
- **arXiv notes：** <https://arxiv.org/abs/2506.02070>
- **入库日期：** 2026-11-07（首次入库 2026-09-21）
- **版本：** 课程站 2026 版（IAP 2026）；官方讲义 PDF 2026-03-18 生成（84 页，作者 Holderrieth、Erives；arXiv:2506.02070 同名）
- **核对范围：** 2026-11-07 抓取课程主页（讲次、实验、讲师、先修要求）与官方讲义 PDF（diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf，84 页）；讲义 §1 生成建模即采样、§2 流模型与扩散模型（ODE/SDE、欧拉与 Euler–Maruyama）、§3 Flow Matching（概率路径、边缘化技巧、CFM 损失与 Theorem 12）、§4 分数与 SDE 采样（Proposition 1、Theorem 17、Fokker–Planck）、§5 引导与 CFG、§6 架构/潜空间/案例（DiT 块、VAE、SD3 与 Movie Gen）、§7 离散扩散与 CTMC 的正文和各 Theorem/Summary 框逐段阅读；证明（附录 B Fokker–Planck 证明、附录 C CTMC 存在唯一性）、附录 A 概率复习、附录 D VAE 补充视角、附录 E 文献指南只浏览；六段讲课录像、幻灯片 PDF 未观看；Lab notebook 未运行，仅核对站点上的实验标题与 Lab 仓库 README 的 changelog。
- **通读：** 已完成
- **一句话说明：** MIT CSAIL **IAP 2026** 生成式 AI 课：从 **ODE/SDE、Fokker–Planck** 到 **flow matching、score matching、CFG、latent diffusion、离散扩散**；三份 Lab 从零搭建 **latent diffusion model**，配套 Colab 与 GitHub 解法。

---

## 为什么值得保留

- **数学–工程闭环：** 不只讲图像生成 demo，而是把 **随机分析工具箱**（SDE 采样、概率路径、向量场）与 **DiT / VAE / CTMC** 组件逐步拆开。
- **与本库交叉：** 机器人侧 [Diffusion Policy](../../wiki/methods/diffusion-policy.md)、[Probability Flow](../../wiki/formalizations/probability-flow.md)、[π₀ flow matching VLA](../../wiki/methods/π0-policy.md) 的上游 **形式化课程**；与 [Sergey Levine Simons 报告](../../wiki/overview/sergey-levine-diffusion-expressive-policies.md) 形成「理论课 ↔ 控制应用」对读。

## 课程结构（2026 版站点）

| Lecture | 主题 |
|---------|------|
| 1 | 生成模型入门；ODE/SDE；flow & diffusion 采样 |
| 2 | Flow Matching：条件/边际概率路径与向量场；FM 训练目标 |
| 3-A | Score function；denoising score matching；SDE 采样 |
| 3-B | Classifier / classifier-free guidance |
| 4 | VAE 潜空间；DiT & U-Net；大规模案例 |
| 5 | 离散扩散；CTMC 采样与训练 |

## Labs（GitHub + Colab）

| Lab | 内容 | 入口 |
|-----|------|------|
| Lab 1 | ODE & SDE 数值实验 | 站点 Lab 1 链接 → GitHub `.ipynb` |
| Lab 2 | Flow matching & score matching 玩具生成模型 | Colab + [lab_two 说明](https://diffusion.csail.mit.edu/2026/labs/lab_two.html) |
| Lab 3 | Diffusion Transformer + VAE；从零 latent diffusion | 站点 Lab 3 |

**解法：** [iap-diffusion-labs `2026` 分支 solutions](https://github.com/eje24/iap-diffusion-labs/tree/2026)

## 引用（站点推荐）

```bibtex
@misc{flowsanddiffusions2026,
  author       = {Peter Holderrieth and Ezra Erives},
  title        = {Introduction to Flow Matching and Diffusion Models},
  year         = {2026},
  url          = {https://diffusion.csail.mit.edu/},
  eprint       = {2506.02070},
  archivePrefix = {arXiv}
}
```

## 对 wiki 的映射

- [mit-flow-matching-diffusion-2026](../../wiki/overview/mit-flow-matching-diffusion-2026.md) — **阅读坐标总览**（本次新建）
- [Probability Flow（形式化）](../../wiki/formalizations/probability-flow.md)
- [Diffusion Policy（方法）](../../wiki/methods/diffusion-policy.md)
- [Diffusion Model（概念）](../../wiki/concepts/diffusion-model.md)
- [sources/repos/iap_diffusion_labs.md](../repos/iap_diffusion_labs.md) — Lab 仓库归档

## 参考来源（原始）

- 课程主页：<https://diffusion.csail.mit.edu/2026/index.html>
- Labs：<https://github.com/eje24/iap-diffusion-labs/tree/2026>

## 通读核对结论（2026-11-07，对照 2026 版官方讲义）

- **讲义目录：** 1 引言（生成即采样）；2 流与扩散模型；3 Flow Matching；4 分数函数与分数匹配；5 引导（Guidance）；6 搭建大规模图像/视频生成器（网络架构、潜空间 VAE、案例 SD3 与 Meta Movie Gen）；7 离散扩散（CTMC，可选）；附录 A 概率论复习、B Fokker–Planck 证明、C CTMC 存在唯一性、D VAE 补充视角、E 扩散文献指南。注意：站点把“Lecture 3-A/3-B”拆成分数匹配与 CFG，讲义里分别对应第 4、5 节。
- **约定：** $t=0$ 为噪声 $p_{init}=\mathcal N(0,I)$，$t=1$ 为数据；流模型 $dX_t=u_t^\theta(X_t)dt$，欧拉法 $X_{t+h}=X_t+hu_t(X_t)$；扩散模型 $dX_t=u_t^\theta dt+\sigma_t dW_t$，Euler–Maruyama $X_{t+h}=X_t+hu_t+\sqrt h\sigma_t\epsilon$；$\sigma_t=0$ 即流模型。
- **Flow Matching：** 条件概率路径 $p_t(\cdot|z)$ 满足 $p_0=p_{init}$、$p_1=\delta_z$；高斯路径 $\mathcal N(\alpha_tz,\beta_t^2I)$，$\alpha_0=\beta_1=0$、$\alpha_1=\beta_0=1$；条件向量场 $u_t(x|z)=(\dot\alpha_t-\frac{\dot\beta_t}{\beta_t}\alpha_t)z+\frac{\dot\beta_t}{\beta_t}x$（式 20）；Theorem 9（边缘化技巧）、Theorem 11（连续性方程）、Theorem 12（$\mathcal L_{FM}=\mathcal L_{CFM}+C$，训练“无需模拟”）；CondOT 路径 $\alpha_t=t,\beta_t=1-t$ 的损失 $\|u^\theta_t(tz+(1-t)\epsilon)-(z-\epsilon)\|^2$（Algorithm 3），讲义称 SD3 与 Meta Movie Gen Video 就用此类目标训练。
- **分数与 SDE：** 高斯路径下向量场与分数可互换（Proposition 1）：$u_t(x)=a_t\nabla\log p_t(x)+b_tx$；Theorem 17（SDE 扩展技巧）：对任意 $\sigma_t\ge0$，$dX_t=[u_t+\frac{\sigma_t^2}{2}\nabla\log p_t]dt+\sigma_tdW_t$ 保持边缘 $p_t$；Fokker–Planck 方程（Theorem 19）；Langevin 动力学是 $p_t$ 固定时的特例；去噪分数匹配损失对 $\beta_t\to0$ 数值不稳定。
- **引导：** 带标签的 flow 模型；CFG：$\tilde u_t(x|y)=(1-w)u_t(x|\varnothing)+wu_t(x|y)$，$w>1$；训练时以概率 $\eta$ 把 $y$ 换成 $\varnothing$；对扩散模型同样适用。
- **架构（§6）：** 时间用傅里叶特征嵌入，类标签用学习嵌入，文本用冻结的 CLIP 等；DiT 基于 ViT（patchify → 线性嵌入 → L 个 DiT 块 → depatchify）；讲义给出的**通用 DiT 块**含自注意力、对提示的交叉注意力与 AdaLN 时间调制，并带可学习门控 $g(\tilde t)$；实验里的类条件 DiT 更简单。与 [DiT 原论文](../papers/peebles_dit_arxiv_2212_09748.md)不同之处：原论文默认的 adaLN-Zero 块没有交叉注意力。VAE 目标 $\mathcal L=\mathcal L_{recon}+\beta D_{KL}(q_\phi(\cdot|x)\|p_{prior})$；潜空间训练后用解码器均值解码；讲义称当下几乎所有最先进的图像/视频生成都走潜扩散范式。
- **案例：** SD3 的 MM-DiT；Movie Gen Video 用 CondOT 目标、时间自编码器（TAE）、三种文本嵌入（UL2、ByT5、MetaCLIP），最大模型 30B 参数。
- **离散扩散（§7）：** 状态空间 $S=V^d$；用 CTMC 取代 SDE，速率矩阵 $Q_t$；Theorem 33（CTMC 存在唯一）；因子化混合路径（逐 token 以概率 $1-\kappa_t$ 破坏）；Theorem 36（离散边缘化技巧）；Kolmogorov 前向方程。
- **站点信息：** 六讲：1 流与扩散模型；2 Flow Matching；3-A 分数函数与分数匹配；3-B CFG；4 潜空间与网络架构（VAE、DiT 与 U-Net、案例）；5 离散扩散。三个 Lab：Lab 1 Working with ODEs and SDEs、Lab 2 Flow Matching and Score Matching、Lab 3 Diffusion Transformer and VAEs。先修：线性代数、多元微积分、基础概率，会 Python/PyTorch。许可：CC BY-NC-SA。
- **原文没有的内容：** 本课与机器人的对应关系（Diffusion Policy、π₀ 等）是本站补充，讲义不含机器人内容。
