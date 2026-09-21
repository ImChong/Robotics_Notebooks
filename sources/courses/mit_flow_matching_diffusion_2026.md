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
- **入库日期：** 2026-09-21
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
