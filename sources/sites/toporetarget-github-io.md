# toporetarget-web（TopoRetarget 项目页）

> 来源归档（ingest）

- **标题：** TopoRetarget — Interaction-Preserving Retargeting for Dexterous Manipulation
- **类型：** site / project-page
- **URL：** <https://tsinghua-mars-lab.github.io/toporetarget-web/>
- **PDF 镜像：** <https://tsinghua-mars-lab.github.io/toporetarget-web/static/images/paper.pdf>
- **入库日期：** 2026-06-17
- **配套论文：** [TopoRetarget（arXiv:2606.16272）](https://arxiv.org/abs/2606.16272) — 归档见 [`sources/papers/toporetarget_arxiv_2606_16272.md`](../papers/toporetarget_arxiv_2606_16272.md)

## 一句话摘要

清华大学 **IIIS / MARS Lab**（Hang Zhao 组）提出的 **TopoRetarget** 官方站点：展示 **交互保留灵巧重定向 → 轻量 PPO 参考跟踪 → Wuji Hand 零样本转笔/魔方重定向** 全链路；提供与 OmniRetarget、DexPilot、Mink、GeoRT 的并排对比，以及跨物体尺度与灵巧手 embodiment 的增广演示。

## 公开信息要点（截至入库日）

- **机构：** IIIS, Tsinghua University（Jielin Wu、Shenzhe Yao、Guanqi He、Xiaohan Liu 等；Hang Zhao† 通讯作者）。
- **采集侧：** 动捕手套采集人手–物体轨迹（页面 Fig. 1A）。
- **方法板块：**
  - **Method** — 骨方向初始化 → 源/机器人 interaction mesh → 拓扑感知 Laplacian 优化
  - **Comparison with Baselines** — Hand-only（vs OmniRetarget、DexPilot）与 Hand-object interaction（vs OmniRetarget、DexPilot、Mink）
  - **Generalization** — 单演示 → 新物体 mesh / 尺度 / 手型（MANO、Wuji、Leap）无需逐例调参
  - **Real-world Results** — Wuji Hand 零样本魔方重定向与转笔；5/5 零样本 trial 保持转笔
- **定量主张（项目页摘要，与论文 Table 1–2 一致）：**
  - ContactPose：接触精度与对齐优于全部基线；相对基线平均 **接触精度误差 −55%**、**最大穿透 −92%**
  - Pen-Spin RL 训练成功率相对既有基线 **+40.6 百分点**
  - 单帧求解 **< 5 ms**，支持实时重定向
- **arXiv 另注项目页：** 论文 HTML 亦链出 <https://toporetarget2026.github.io/TopoRetarget/>（与 MARS Lab Pages 镜像并存，以用户指定 URL 为主归档）。

## 为何值得保留

- **非 PDF 证据：** 真机转笔长镜头、增广 hover 对比与基线 artifact 图比摘要更直观呈现 **interaction mesh** 在灵巧手场景的收益。
- **与 OmniRetarget 互证：** 同属 **Laplacian interaction mesh** 族，但面向 **dexterous hand–object** 而非人形全身 loco-manipulation，便于跨尺度对照阅读。
- **下游硬件锚点：** Wuji Hand 零样本 sim2real 为 [舞肌科技 Wuji Hand](../../wiki/entities/wuji-robotics.md) 提供近期 contact-rich 参考跟踪案例。

## 关联资料

- 论文归档：[`sources/papers/toporetarget_arxiv_2606_16272.md`](../papers/toporetarget_arxiv_2606_16272.md)
- 近邻全身交互保留：[`sources/papers/omniretarget_arxiv_2509_26633.md`](../papers/omniretarget_arxiv_2509_26633.md)
- 物理采样灵巧重定向：[`sources/papers/spider_scalable_physics_informed_dexterous_retargeting.md`](../papers/spider_scalable_physics_informed_dexterous_retargeting.md)

## 对 wiki 的映射

- [`wiki/methods/toporetarget-interaction-preserving-dexterous-retargeting.md`](../../wiki/methods/toporetarget-interaction-preserving-dexterous-retargeting.md)
