# DeWorldSG: Depth-Aware 3D Semantic Scene Graph Generation via World-Model Priors（arXiv:2607.00889）

> 来源归档（ingest）

- **标题：** DeWorldSG: Depth-Aware 3D Semantic Scene Graph Generation via World-Model Priors
- **类型：** paper / 3D semantic scene graph / RGB-D / world-model priors
- **arXiv：** <https://arxiv.org/abs/2607.00889>（PDF：<https://arxiv.org/pdf/2607.00889.pdf>；HTML：<https://arxiv.org/html/2607.00889>）
- **项目页：** <https://deworldsg2026.github.io/>
- **会议：** ECCV 2026（Malmö, Sweden）
- **作者：** Seok-Young Kim、Abdelrahman Elskhawy、Taewook Ha、Dooyoung Kim、Eunjae Shin、Benjamin Busam†、Woontack Woo†
- **机构：** 韩国科学技术院（KAIST）；慕尼黑工业大学（TU Munich）；慕尼黑机器学习中心（MCML, Munich Center for Machine Learning）
- **入库日期：** 2026-09-11
- **一句话说明：** 从 RGB-D 序列增量生成时空一致的 **3D 语义场景图**：实例级 **深度感知 3D 高斯节点** + **Dual-Domain Depth Refinement**，关系侧用跨帧证据聚合并融合 **V-JEPA 2** 世界模型先验，缓解逐帧推理的关系稀疏与几何不稳。

## 开源状态（项目页核查，2026-09-11）

- **论文 / arXiv 摘要：** 写 *Our code and models are open-sourced*。
- **项目页：** 页首 Code 按钮为 **Coming Soon**（GitHub 图标链接 disabled）；**未列** 训练/推理仓库 URL。
- **GitHub 检索：** 仅见 [`deworldsg2026/deworldsg2026.github.io`](https://github.com/deworldsg2026/deworldsg2026.github.io)（项目页静态站），**未见** 官方算法/权重仓库。
- **结论：** **宣称开源 / 待发布** — 以项目页实际链接为准；入库日不可复现，后续 lint 可跟进。

## 摘要级要点

- **问题：** 现有 3D SSG 方法常把物体压成单点投影、逐帧推断关系 → 几何不稳、关系稀疏、时序不一致。
- **物体节点：** SAM 实例 mask + 深度引导滤波 → 实例级 **3D 高斯分布** \((\mu_i^{3D}, \Sigma_i^{3D})\) 而非单点。
- **Dual-Domain Depth Refinement（DR）：** 空间域 + 深度域联合去噪，抑制 flying pixels，稳定全局 merge。
- **全局图：** 语义一致性 + 高斯 Hellinger 相似度增量 merge 局部 3D 子图。
- **关系 refine：** 对不确定边聚合 16 帧 union-crop clip 证据；冻结 **V-JEPA 2** + 轻量 MLP probe 产出 \(p_{\mathrm{WM}}(r_{i\to j})\)，与几何证据融合。
- **评测：** 3DSSG、ReplicaSSG；相对 prior SoTA（FROSS）Relation Recall **+77.4%**、Object **+20.2%**、Predicate **+23.2%**（3DSSG）；ORB-SLAM3 位姿下仍保留 GT 的 **89.2%** Obj. R / **87.6%** Rel. R。
- **延迟：** ReplicaSSG 四场景平均 **108.53 ms/帧**（SAM 25.26 ms + 关系 refine 69.39 ms 为主开销）。

## 核心论文摘录（MVP）

### 1) 概率 3D 节点 + Dual-Domain Depth Refinement

- **链接：** §3.1–3.2；Eq. (1)–(2)
- **摘录要点：** 每实例从 mask 内深度采样建 3D 高斯；DR 在空间/深度双域滤波 flying pixels 与噪声，再用于全局 merge 的 \(\delta_g=0.7\) Hellinger 阈值匹配。
- **对 wiki 的映射：**
  - [DeWorldSG](../../wiki/entities/paper-deworldsg.md) — 几何 lifting 核心。
  - [vS-Graphs](../../wiki/entities/paper-vs-graphs-visual-slam-scene-graph.md) — 几何优先 3D 场景图 SLAM 对照。

### 2) 增量全局 merge + 时空关系聚合

- **链接：** §3.3–3.4
- **摘录要点：** 局部 3D 子图按语义 + 高斯相似 merge；对熵高于阈值的不确定边，跨帧累积 predicate 分布后再注入全局图。
- **对 wiki 的映射：**
  - [DeWorldSG](../../wiki/entities/paper-deworldsg.md) — 流程总览。
  - [Functional-SLAM](../../wiki/entities/paper-functional-slam.md) — 在线维护 3D 场景图的另一路线。

### 3) V-JEPA 2 世界模型关系先验

- **链接：** §3.5；Fig. 3；Tab. 3 (e) vs (d)
- **摘录要点：** 16 帧 union-crop → 冻结 V-JEPA 2 → mean-pool → MLP probe；相对 DINOv2 静态表征，时序 WM 先验带来更高 Relation / Predicate Recall（Tab. 3：Rel. 50.2 vs 49.5，Pred. 57.3 vs 56.6）。
- **对 wiki 的映射：**
  - [DeWorldSG](../../wiki/entities/paper-deworldsg.md) — WM 先验模块。
  - [V-JEPA 2](../../wiki/entities/paper-vjepa2.md) — 被引用的预训练世界模型。

### 4) 3DSSG / ReplicaSSG 量化与 ablation

- **链接：** §5；Tab. 1–3
- **摘录要点：** 3DSSG Rel./Obj./Pred. **50.2 / 75.0 / 57.3**（FROSS 27.9 / 62.4 / 33.0）；ReplicaSSG GT pose **38.2 / 35.4 / 45.3**；mask + DR + WM 逐步 ablation 均单调增益。
- **对 wiki 的映射：**
  - [DeWorldSG](../../wiki/entities/paper-deworldsg.md) — 评测节。
  - [导航与 SLAM 自主栈](../../wiki/overview/navigation-slam-autonomy-stack.md) — 场景图在机器人栈中的位置。

## BibTeX

```bibtex
@inproceedings{kim2026deworldsg,
  title={DeWorldSG: Depth-Aware 3D Semantic Scene Graph Generation via World-Model Priors},
  author={Kim, Seok-Young and Elskhawy, Abdelrahman and Ha, Taewook and Kim, Dooyoung and Shin, Eunjae and Busam, Benjamin and Woo, Woontack},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-deworldsg.md`](../../wiki/entities/paper-deworldsg.md)
- 项目页：[`sources/sites/deworldsg-website.md`](../sites/deworldsg-website.md)
- 互链：[V-JEPA 2](../../wiki/entities/paper-vjepa2.md)、[Functional-SLAM](../../wiki/entities/paper-functional-slam.md)、[SayPlan](../../wiki/entities/paper-sayplan-llm-scene-graph-planning.md)、[导航与 SLAM 自主栈](../../wiki/overview/navigation-slam-autonomy-stack.md)
