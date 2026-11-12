# Stochastic Interpolants：统一 Flows 与 Diffusions 的随机插值框架

- **英文题目：** Stochastic Interpolants: A Unifying Framework for Flows and Diffusions
- **作者：** Michael S. Albergo、Nicholas M. Boffi、Eric Vanden-Eijnden
- **出处：** Journal of Machine Learning Research 26(209):1–80, 2025
- **期刊一手页面：** https://www.jmlr.org/papers/v26/23-1605.html
- **PDF：** https://www.jmlr.org/papers/volume26/23-1605/23-1605.pdf
- **归档日期：** 2026-11-12
- **核对范围：** 期刊正式题录、摘要、原文目录及核心框架定义；80 页全文的证明、实验与附录未逐页核查。
- **通读：** 未完成

## 核心摘录

- 通过随机插值在有限时间连接任意两给定概率密度，导出 transport equation、具有可调扩散系数的 Fokker–Planck 方程。
- 同一个框架对应 deterministic probability flow / ODE 和 stochastic diffusion / SDE；漂移速度与 score 可以经二次目标学习。
- 文中还分析似然控制、cross-entropy 估计，以及 Schrödinger bridge 联系。
- 对 4D-HOF 的作用：提供把一个有结构的初始化状态向目标交互分布搬运的理论基础；4D-HOF 的具体网络、HOI 损失与 guidance 是其自身设计，不能归于本论文。

## 对 wiki 的映射

- [4D-HOF 论文实体](../../wiki/entities/paper-4d-hof.md) — 作为随机插值与条件桥的直接理论引用。
- 此条目前保留原始资料归档；完整理论推导深读后再考虑独立 formalization 页。
