# RigMo — 项目页

- **来源（推荐）：** <https://haoz19.github.io/RigMo-page/>
- **镜像：** <https://rigmo-page.github.io/>（截至 2026-07-23 仍显示 Code/Data Coming Soon）
- **类型：** site（项目页）
- **机构：** Snap Inc. · UIUC · UC Santa Cruz · CMU · NTU
- **归档日期：** 2026-07-23
- **论文：** arXiv:2601.06378 · [`sources/papers/rigmo_arxiv_2601_06378.md`](../papers/rigmo_arxiv_2601_06378.md)

## 一句话说明

**RigMo** 是从原始变形 mesh 序列 **联合学习 rig 结构与运动动力学** 的生成式框架：无需 GT 骨架/蒙皮或逐序列优化，输出可动画的 **Gaussian bones + SE(3) 运动**，并在结构感知潜空间上支持 **Motion-DiT** 补全/生成。

## 为什么值得保留

- 项目页给出方法图（RigMo-VAE 双路径、Motion-DiT）、跨人类/动物/非人形形状的定性结果，以及 Paper / Code / Data 入口。
- **开源入口以作者个人项目页为准**：Code → GitHub；Data → Hugging Face gated dataset。
- 与 Disney「Generative Motion Rig」（DCC 插件工作流）名称相近但问题完全不同——本页锚定 **无标注 mesh→可动画资产** 研究线。

## 开源状态（2026-07-23 核查）

| 项 | 结论 |
|----|------|
| Code | <https://github.com/haoz19/RigMo> — **RigMo-VAE** 可训练；**Motion-DiT 未包含** |
| Data | <https://huggingface.co/datasets/haoz19/RigMo-data> — gated，~2 万序列预处理 |
| 镜像页 | `rigmo-page.github.io` 文案滞后，勿仅凭其「Coming Soon」判定未开源 |

## 对 wiki 的映射

1. **[RigMo（实体页）](../../wiki/entities/rigmo.md)** — 架构、开源边界、源码运行时序图
2. **[Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md)** — Motion-DiT / 结构感知潜空间
3. **[Generative Motion Rig（Disney）](../../wiki/entities/generative-motion-rig.md)** — 名称相近的 DCC 工作流对照（勿混淆）
