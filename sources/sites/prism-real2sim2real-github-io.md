# prism-real2sim2real.github.io

> 来源归档（项目站点）

- **标题：** PRISM — Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation
- **类型：** site
- **URL：** <https://prism-real2sim2real.github.io/>
- **论文：** [arXiv:2609.38172](https://arxiv.org/abs/2609.38172)
- **会议：** CoRL 2026
- **机构：** Amazon FAR；UC Berkeley；Stanford University；Carnegie Mellon University
- **代码：** <https://github.com/amazon-far/PRISM-Real2Sim2Real>（站点 Header **Code** 按钮；截至 **2026-09-30** 仓库 HTTP **404**）
- **入库日期：** 2026-09-30
- **最近复核：** 2026-09-30
- **一句话说明：** 项目页展示 V2V counterfactual 扩数据、Real2Sim 管线、G1 真机 pick–carry–drop 与 pose/scale/robustness 视频；PDF 与 BibTeX 链自站点。

## 站点要点（ingest 摘录）

1. **tl;dr：** 4 条真人视频 → 多样化 loco-manipulation 训练数据。
2. **部署约束：** 仅机载深度；单策略 + 摇杆；**零样本 Sim2Real**、无 MOCAP。
3. **V2V prompt 模板：** 保留背景/光照/机位；`replace the box with <CLS>, pick it up and carry with two hands`；`<CLS>` ∈ {box, bin, barrel, ball}。
4. **管线动画：** CF video → 3D 重建 → 重定向 → 仿真学习 → 真机。
5. **Robustness：** 137 生成 clip 相对 4 seed 的物体位姿/yaw 覆盖；平地训策略零样本 **35° 坡** 与 **0.43 m** 台架 pick-up（45°/0.45 m 失败）。

## 交叉索引

- 论文归档：[prism_real2sim2real_arxiv_2609_38172.md](../papers/prism_real2sim2real_arxiv_2609_38172.md)
- Wiki 实体：[paper-prism-real2sim2real.md](../../wiki/entities/paper-prism-real2sim2real.md)
- 代码归档（待仓公开后补 README）：[prism-real2sim2real.md](../repos/prism-real2sim2real.md)
