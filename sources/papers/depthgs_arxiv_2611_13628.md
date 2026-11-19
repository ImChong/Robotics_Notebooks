# DepthGS：深度引导的高斯溅射与自适应正则化表面重建

> 论文一手资料逐节核对归档；只摘录机制、验证结论与复现边界，不转载原文。

- **类型：** paper
- **标题（英文）：** DepthGS: Depth-Guided Gaussian Splatting with Adaptive Regularization for High-Quality Surface Reconstruction
- **标题（中文）：** DepthGS：深度引导的高斯溅射与自适应正则化高质量表面重建
- **作者：** Xiao Ren, Yu Liu, Ning An, Jian Cheng, Xin Qiao, You-Fu Li, He Kong
- **机构：** 南方科技大学；中煤科工矿山人工智能研究院／煤矿智能开采与岩层控制全国重点实验室；西安交通大学；香港城市大学
- **期刊：** IEEE Transactions on Pattern Analysis and Machine Intelligence（TPAMI；2026-11-06 online ahead of print）
- **DOI：** https://doi.org/10.1109/TPAMI.2026.3751295
- **论文：** https://arxiv.org/abs/2611.13628
- **完整 HTML：** https://arxiv.org/html/2611.13628v1
- **PDF：** https://arxiv.org/pdf/2611.13628
- **原文版本：** arXiv:2611.13628v1，2026-11-10 提交；HTML 首屏记录版本
- **代码：** 截至 2026-11-19，论文及 arXiv 页面未提供可核实的官方 DepthGS 代码仓库入口；不能将所依赖的 PGSR 代码等同于本文实现
- **项目主页：** 未发现可核实的独立官方项目页（截至 2026-11-19）
- **硬件：** 作者全部实验使用单张 NVIDIA GeForce RTX 4090
- **入库日期：** 2026-11-19
- **通读：** 已完成
- **核对范围：** 原文 arXiv HTML v1：摘要、引言、Related Work、3DGS 基础、第 IV 节单视角与多视角误差推导／正则化／总损失、第 V 节实现设置及 Tables I–V／消融与优化动态、第 VI 节结论、文末参考文献；论文中的可见图片说明和限制讨论均已核对。
- **沉淀到 wiki：** [DepthGS 独立论文实体](../../wiki/entities/paper-depthgs.md)
- **相关纵深路线：** [Real2Sim Stage 1](../../roadmap/depth-real2sim.md)

## 核心摘录

### 问题与理论出发点

- 标准 3DGS 渲染逼真，但不规则高斯原语未必准确拟合实体表面；本文基于 PGSR 的无偏深度 `D_u` 与渲染距离 `D_𝒟`，定义逐像素深度差 `ΔD = D_u - D_𝒟`。
- **单视角 Lemma 1：** 利用三角不等式得到重建误差上界 `ε_u ≤ ε_𝒟 + |ΔD|`。这个上界并不意味着单凭 `|ΔD|` 就能测到真实几何误差：还含未知真实表面的 `ε_𝒟` 项。
- **多视角 Lemma 2：** 跨视角基于渲染距离的重投影误差与深度差向量残差共同给出无偏深度点云的误差上界；以两种深度的跨视角残差构造实际可优化的正则项。完整推导见论文 §IV-B。

### 两个新机制

1. **单视角自适应法线正则：** 对 `|ΔD|` 做 min-max 归一化并取反得到 `W_ΔD`；局部深度差小则加大渲染法线与由无偏深度生成的法线之间约束，深度差大则降低权重，以免不可靠的局部几何主导优化。另保留 PGSR 的图像边缘权重。
2. **跨视角深度一致性：** 对相邻相机之间的 `D_u` 和 `D_𝒟` 分别进行坐标变换、重投影与双线性采样，构造 `δ_u` 与 `δ_𝒟`；有效像素掩码剔除越界或深度小于阈值的对应，`L_mv_D = mean(α δ_u + (1-α) δ_𝒟)`，其中 `α = 0.8`。
3. **总目标：** RGB 重建、Gaussian flattening、多视角光度项（PGSR 原有）+ 自适应单视角法线项 + 新增跨视角深度项；训练后进行 TSDF 融合与网格提取。

### 实验与边界（原文表格）

| 数据集 / 指标 | PGSR 或对照 | DepthGS | 解释 |
| --- | --- | --- | --- |
| DTU 平均 Chamfer Distance（mm，越低越好） | PGSR 0.53；GSPrior 0.50 | **0.51** | 优于 PGSR，但该表中非第一 |
| DTU 优化时间 | PGSR 17 min | **17.6 min** | 接近基线 |
| Tanks and Temples 平均 F1（越高越好） | PGSR 0.495；GSPrior 0.507 | **0.503** | 竞争性结果，非第一 |
| Mip-NeRF360 Overall PSNR（dB，越高越好） | PGSR 27.45 | **27.72** | 相对基线提升 |
| Mip-NeRF360 Overall SSIM（越高越好） | PGSR 0.841 | **0.841** | 持平 |
| Mip-NeRF360 Overall LPIPS（越低越好） | PGSR 0.181 | **0.175** | 感知指标改善 |
| DTU 消融：只用自适应法线 / 完整模型 CD | 0.51 / 0.51 | **0.51** | 跨视角项未进一步降低该聚合 CD，但论文给出 E_MVC 改善证据 |
| DTU 完整模型 GPU 显存（GB） | PGSR avg 6.41 / peak 7.12 | **avg 6.77 / peak 7.50** | 有显存开销 |

- 作者使用单张 RTX 4090、每场景 30,000 次迭代、`λ5 = 0.008`、`τ_d = 0.1`、`σ = 10⁻⁶`；除新组件外沿用 PGSR 超参数。
- 优化曲线并非所有场景都一致提升：论文报告 bicycle 场景 test-view PSNR 相对 PGSR 下降 0.05 dB。
- 论文结论指出复杂户外强光照、噪声几何及无结构场景仍有鲁棒性挑战；没有公开实现供独立复现，不能把单卡 RTX 4090 的论文训练时间推断成机器人实时重建能力。
- **机器人使用边界：** 产物是静态几何重建 / RGB 新视角渲染和 TSDF 网格；将其变成 sim-ready 资产还需碰撞、摩擦、质量、关节与动态感知建模。

## 对 wiki 的映射

- [DepthGS：深度差加权几何优化](../../wiki/entities/paper-depthgs.md) — 独立、非重复论文节点。
- [六种空间表征](../../wiki/concepts/embodied-perception-six-spatial-representations.md) — 深度、3DGS 表征与 TSDF 网格的区别。
- [Real2Sim Stage 1](../../roadmap/depth-real2sim.md) — 将它放到几何重建环节，不混为物理可用资产。

## 参考来源

- [作者论文全文（arXiv HTML v1）](https://arxiv.org/html/2611.13628v1)
- [arXiv 摘要和版本历史](https://arxiv.org/abs/2611.13628)
- [TPAMI 期刊标识 DOI](https://doi.org/10.1109/TPAMI.2026.3751295)
