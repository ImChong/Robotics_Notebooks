# swap-parkour.github.io（SWAP 项目页）

- **标题：** SWAP: Symmetric Equivariant World-Model for Agile Robot Parkour — 官方项目页
- **类型：** site / project-page
- **URL：** <https://swap-parkour.github.io/>
- **入库日期：** 2026-06-19
- **配套论文：** [SWAP（arXiv:2606.19928）](https://arxiv.org/abs/2606.19928) — 归档见 [`sources/papers/swap_parkour_arxiv_2606_19928.md`](../papers/swap_parkour_arxiv_2606_19928.md)

## 一句话摘要

浙江大学与 Mirrorme 提出的 **SWAP** 官方站点：展示 **对称等变潜变量世界模型 + 等变 Actor-Critic** 端到端四足跑酷框架、**2.13 m 远跳 / 1.63 m 攀台** 实机纪录，以及高动态机动、恶劣环境适应与户外零样本泛化视频。

## 公开信息要点（截至入库日）

- **机构：** 浙江大学 X-Mechanics、ZJU-Hangzhou Global Scientific and Technology Innovation Center、Mirrorme Technology Co., Ltd.（* equal；† Yongbin Jin & Hongtao Wang 通讯作者）。
- **方法概览（主页）：** 低频 **Symmetric Equivariant World Model** 将镜像物理观测映射为镜像潜状态；高频 **Equivariant Actor** 生成对称动作，**Invariant Critic** 保证镜像状态价值一致。
- **演示板块：**
  - **Extreme Locomotion** — 极限跑酷能力
  - **Hike: Highly Agile Maneuvers** — 高动态机动
  - **Hike: Adaptation to Adverse Environments** — 恶劣环境适应
- **摘要（与 arXiv 一致）：** 打破四足跑酷纪录；对未见镜像地形鲁棒几何泛化；多样户外环境 exceptional zero-shot transfer。
- **BibTeX：** `@article{lan2026swap, ... journal={arXiv preprint}, year={2026}}`

## 为何值得保留

- **非 PDF 证据：** 视频比静态论文更直观呈现远跳蹬地、双侧协调攀台与户外扰动下的闭环行为。
- **与 arXiv 三角互证：** 摘要、方法图（WM 蓝 / Policy 紫 双频架构）与站点演示板块一致，便于 lint 核对表述。
- **跑酷技术谱系锚点：** 在 Extreme Parkour / WMP 之后，把 **对称结构先验 × 世界模型** 推到四足跑酷性能前沿。

## 关联资料

- 论文归档：[`sources/papers/swap_parkour_arxiv_2606_19928.md`](../papers/swap_parkour_arxiv_2606_19928.md)
- Wiki 实体：[`wiki/entities/paper-swap-parkour.md`](../../wiki/entities/paper-swap-parkour.md)
