# Geometry and Gradient-based Partitioning for Panoramic Outdoor Reconstruction（PanoLOG / G²PS）

> 来源归档（ingest）

- **标题：** Geometry and Gradient-based Partitioning for Panoramic Outdoor Reconstruction
- **缩写：** **PanoLOG**（框架）/ **G²PS** · **G2PS** · **GGPS**（Geometry and Gradient-based Partitioning Strategy；仓库名）
- **类型：** paper / 3dgs / panoramic / outdoor-reconstruction / novel-view-synthesis / dataset
- **arXiv：** <https://arxiv.org/abs/2607.08769>（PDF: <https://arxiv.org/pdf/2607.08769>）
- **HTML：** <https://arxiv.org/html/2607.08769>
- **项目页：** <https://insta360-research-team.github.io/GGPS-Website/>
- **代码：** <https://github.com/Insta360-Research-Team/GGPS>（完整训练代码已发布）
- **Hugging Face：** <https://huggingface.co/Insta360-Research/GGPS>（Pano360 部分场景 + `.ply` 占位）
- **机构：** 影石研究（Insta360 Research）；中山大学（SYSU）；华南理工大学（SCUT）；中国科学院大学（UCAS）；哈尔滨工程大学（HEU）；武汉大学（WHU）
- **作者：** Weijian Chen、Weibo Yao、Yuhang Zhang、Xiaolin Tang、Guo Wang、Weijun Zhang、Xitong Gao、Yihao Chen、Hongde Qin、Lu Qi（通讯）
- **状态：** arXiv 预印本（约 2026-07）；**已开源（训练代码 + 部分数据集）** — 预训练 `.ply` 与 UE 5.8 3DGS 插件仍待发布
- **许可：** CC BY-NC 4.0（非商业）
- **入库日期：** 2026-07-26
- **一句话说明：** 面向 **ERP 全景户外大场景 3DGS**：两阶段粗到细框架 **PanoLOG** + **G²PS**（视差不确定度扩 AABB + 梯度重要性相机–块分配），解决 360° 全可见性导致块划分退化为全局训练的问题；并发布 **Pano360** 基准。

## 摘录 1：问题与贡献

- **动机：** 窄 FoV 大规模采集成本高；改用 **ERP 全景** 可显著减采集量，但 **全向可见性** 使依赖针孔 frustum / 局部可见性的划分（CityGaussian、VastGaussian、H3DGS 等）失去判别力，块并行退化为全局优化。
- **框架：** **PanoLOG** — Stage I 全局粗训（天空球 + 全景单目深度先验）→ Stage II **G²PS** 划分后块并行精炼 → 合并。
- **贡献（论文三条）：**
  1. **PanoLOG**：全景大场景 3DGS 的粗到细管线，把无效全局优化转为可扩展块并行。
  2. **G²PS**：几何（视差驱动的深度不确定度 → 自适应 AABB）+ 梯度重要性相机–块分配。
  3. **Pano360**：首个大规模户外全景重建基准（论文口径 4 场景、5637 张 3840×1920）。
- **关键词：** 3DGS、ERP、partitioning、omnipresent visibility、Pano360、novel view synthesis。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-panolog-ggps.md`](../../wiki/entities/paper-panolog-ggps.md)；交叉 [Glob3R](../../wiki/entities/paper-glob3r.md)、[PanoWorld](../../wiki/entities/paper-panoworld-real-world-panoramic-generation.md)、[GS-Playground](../../wiki/entities/gs-playground.md)、[导航·SLAM 栈](../../wiki/overview/navigation-slam-autonomy-stack.md)。

## 摘录 2：基表示（ERP 3DGS + 天空球 + 深度）

- **渲染：** 各向异性 3D 高斯 + ERP 投影；2D 协方差经 ERP Jacobian（非透视投影矩阵）。
- **显式天空球：** 在 $R_{\mathrm{sky}}=\kappa\cdot r_{\mathrm{scene}}$（$\kappa=10$）上初始化 $N_{\mathrm{sky}}=10^5$ 天空高斯；粗训时位置梯度置零、排除 densification；精炼阶段 **整球冻结**，保证跨块天空一致。
- **全景深度监督：** DAP（Lin et al., 2026）产 ERP 单目逆深度；与 SfM 稀疏深度按图仿射对齐；损失在 **径向深度** $\|\mathbf{X}\|$ 上，权重指数衰减。
- **统一损失：** $(1-\lambda_{\mathrm{ssim}})\mathcal{L}_1 + \lambda_{\mathrm{ssim}}\mathcal{L}_{\mathrm{D\text{-}SSIM}} + \mathcal{L}_{\mathrm{depth}}$。

**对 wiki 的映射：** 实体页「核心原理 / 流程总览」。

## 摘录 3：G²PS（几何划分 + 梯度分配）

- **几何 AABB：** 相机轨迹轴半径 $r_d$ + 代表基线 $\hat{b}$（最近邻距离中位数）× 三角化范围因子 $\rho_{\mathrm{tri}}$（默认 5）得 margin；收缩后均匀分块，避免无界户外空块失衡。
- **梯度相机–块分配：** Stage I 收敛后，对每视角做一次前向–反向，块内位置梯度均值 $s_{k,b}$；相机归属块当 **几何落在块内** 或 **归一化梯度比 $>\tau_{\mathrm{grad}}$（默认 0.8）**。
- **块优化与合并：** 各块继承完整粗模型；周期性 opacity reset 后仅本块观测充分的高斯恢复不透明并保留；半开区间保证无重复；邻块共享相机集合 → 边界无需显式缝合。

**对 wiki 的映射：** 实体页方法节与源码时序（`data_partition.py` ↔ `train_large.py --block_id`）。

## 摘录 4：实验要点

- **Pano360 子集：** A1 无人机 NSC / NSK；X5 手持 BAX / NSN；另评 Ricoh360、360Roam。
- **对照：** 大规模针孔系 H3DGS / CityGaussian / DOGS / Momentum-GS（ERP→六面立方体）；全景系 OmniGS / ODGS / SpaGS / 3DGS。
- **主结果（论文表）：** NSC PSNR **28.18**（次优 H3DGS 27.78），模型 **463 MB** vs H3DGS 1002 MB；BAX / NSN 相对最强基线约 **+0.64 / +1.16 dB**；Ricoh360 / 360Roam 全面最优。
- **消融：** 去掉 G2PS ≈ **−0.51 dB**；无天空球易 floaters；无深度削弱极区/远景几何。
- **硬件：** 单卡 RTX 4090 24 GB；各方法 30k iter。

**对 wiki 的映射：** 实体页评测表。

## 摘录 5：开源边界（项目页 + 仓 + HF，截至 2026-07-26）

| 项 | 状态 |
|----|------|
| **训练 / 划分 / 合并 / 渲染 / 评测代码** | **已发布**（`train_large.py`、`data_partition.py`、`merge.py`、`render_large.py`、`metrics_large.py`、`scripts_new/`） |
| **Pano360 数据（HF）** | **部分：** `FTP.zip`、`NSC.zip`、`NSK.zip`（共 2792 张）；论文中的 **BAX / NSN** 尚未见 HF 归档 |
| **预训练 `.ply`** | HF `ply/` **占位**；README 计划 **2026-07 下旬** 放两景复现模型 |
| **UE 5.8 3DGS 渲染插件** | 项目页 / 仓 Roadmap：**即将发布**（免费水印版待审） |
| **许可** | **CC BY-NC 4.0** |
| **结论** | **已开源（可跑训练管线）+ 数据集部分 / 权重与 UE 插件待齐** |

**对 wiki 的映射：** [`sources/sites/insta360-research-team-ggps-website.md`](../sites/insta360-research-team-ggps-website.md)、[`sources/repos/ggps.md`](../repos/ggps.md)；实体页开源状态与源码运行时序图。

## 当前提炼状态

- [x] arXiv HTML / 项目页 / GitHub / Hugging Face 已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-panolog-ggps.md` 新建
- [x] 开源边界写入 sites / repos / wiki（训练已开；数据部分；`.ply`/UE 待发）
