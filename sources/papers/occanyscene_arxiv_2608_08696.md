# occanyscene_arxiv_2608_08696

> 来源归档（ingest）

- **标题：** OccAnyScene: Towards Unified Indoor-Outdoor 3D Occupancy Prediction
- **短名：** OccAnyScene
- **类型：** paper
- **来源：** arXiv abs / PDF
- **原始链接：**
  - <https://arxiv.org/abs/2608.08696>
  - <https://arxiv.org/pdf/2608.08696>
- **项目页：** <https://roboperception.github.io/OccAnyScene/> — 归档见 [`sources/sites/roboperception-occanyscene-github-io.md`](../sites/roboperception-occanyscene-github-io.md)
- **代码仓（占位）：** <https://github.com/RoboPerception/OccAnyScene> — [`sources/repos/occanyscene.md`](../repos/occanyscene.md)
- **作者：** Junjie Liu<sup>1,*</sup>, Wanshui Gan<sup>2,*</sup>, Zitong Dai<sup>3</sup>, Guiping Cao<sup>4</sup>, Yan Li<sup>4</sup>, Ke Chen<sup>4</sup>, Dongmei Jiang<sup>4</sup>, Xiangyuan Lan<sup>4</sup>, Jianguo Zhang<sup>1</sup>（* 同等贡献）
- **机构：** 南方科技大学（SUSTech）；上海人工智能实验室（Shanghai AI Lab）；哈尔滨工业大学深圳（HITSZ）；鹏城实验室（Peng Cheng Laboratory）
- **版本：** arXiv:2608.08696（2026-08）
- **入库日期：** 2026-08-13
- **一句话说明：** 提出 **Cross-Scene 3D Semantic Occupancy Prediction**：同一模型在室内单目与户外六相机、不同空间范围 / 体素 / 语义分类下联合预测；用像素视锥高斯（PFFA + FPGC）做场景自适应 image-to-3D lifting。官方仓截至入库日仅项目材料，**训练/推理待录用后发布**。

## 核心摘录

### 1) 问题与任务设定
- 现有 3D 语义占据方法多为 **scene-specific**：相机配置、空间范围、体素分辨率、语义分类绑死单一室内或驾驶协议。
- 自动驾驶 / 具身系统常需 **同一套感知** 覆盖开放道路（远距粗粒度）与室内/停车场（近距细粒度）；维护多模型难扩展。
- 本文定义 **Cross-Scene 3D Semantic Occupancy Prediction**：单模型在各域 **原生协议** 下联合训练与预测，而不是统一到同一网格/分类再训。
- 难点不只是室内外外观域差，而是 **度量一致、又随相机与场景尺度自适应** 的 image-to-3D lifting：稠密体素绑死预定范围；绝对米制高斯偏移/尺度在室内外分布差一个数量级。

### 2) 方法要点
1. **连续高斯表示：** 每个高斯 \(\mathbf{a}_i=[\boldsymbol{\mu}_i,\mathbf{s}_i,\mathbf{r}_i,o_i,\mathbf{f}_i]\)；共享语义特征 \(\mathbf{f}_i\in\mathbb{R}^{C_f}\)（文中 \(C_f=32\)），再用域特定 taxonomy 矩阵 \(\mathbf{T}_s\) 映射到 \(\mathcal{C}_s\)。
2. **深度基础模型骨干：** Depth Anything V2（ViT-B，默认）或 V3（ViT-L）；ViT tokens + DPT 解码的稠密几何图 \(\mathbf{F}_{\mathrm{geo}}\)（下采样到输入 \(1/8\)）。
3. **PFFA（Pixel-Aligned Frustum Feature Aggregation）：** 像素查询 \(=\phi_{\mathrm{geo}}(\mathbf{F}_{\mathrm{geo}}(p))+\phi_{\mathrm{cam}}(\mathbf{R}(p))\)，再对 ViT token 网格做像素对齐 deformable cross-attention，得到视锥 query。
4. **FPGC（Frustum-Parameterized Gaussian Construction）：** 每像素解码 \(K=3\) 个高斯：表面相对深度增量 \(\Delta d\)、亚像素偏移 \(\Delta\mathbf{u}\)、相对尺度 \(\widehat{\mathbf{s}}\)。
5. **Canonical-camera 深度：** 按 Metric3D v2，先在规范相机空间预测深度再按焦距比还原米制表面深度，作为 \(K\) 个高斯的共同锚点。
6. **视锥相对尺度：** \(b_{p,k}=\eta\cdot\frac12(d/f_x+d/f_y)\)，\(\mathbf{s}=b\cdot\widehat{\mathbf{s}}\)，避免直接回归跨场景绝对尺度。
7. **Gaussian-to-Occ：** 沿用 SplatSSC 的 Decoupled Gaussian Aggregator；损失 \(\mathcal{L}_{\mathrm{occ}}=\lambda_{\mathrm{focal}}\mathcal{L}_{\mathrm{focal}}+\lambda_{\mathrm{lov}}\mathcal{L}_{\mathrm{lov}}+\lambda_{\mathrm{scal}}\mathcal{L}_{\mathrm{scal}}^{\mathrm{prob}}\)，再加 Huber 深度监督，**端到端单阶段**（对比 SplatSSC 两阶段）。
8. **跨场景训练：** 交替数据集、各域有效迭代数与单域设定对齐；**仅 taxonomy 矩阵按数据集分开**，其余权重共享。
9. **相机间隙补全（局限节）：** 像素视锥只覆盖 FOV 内；SurroundOcc 相邻相机间隙用少量可学习空间 query + 全局池化交叉注意力补高斯，对总指标影响小。

### 3) 实验（论文报告摘要）

| 数据集 | 场景 | 输入 | 样本 | 范围 | 体素 |
|--------|------|------|------|------|------|
| Occ-ScanNet | 室内 | 单目 | 47.5K | \(4.8\times4.8\times2.88\) m | 0.08 m |
| SurroundOcc-nuScenes | 户外 | 六相机 | 34.1K | \(100\times100\times8\) m | 0.5 m |

| 设定 / 变体 | Occ-ScanNet IoU / mIoU | SurroundOcc IoU / mIoU | 读法 |
|-------------|------------------------|------------------------|------|
| OccAnyScene-DAv3 scene-specific | **68.34 / 59.92** | **35.97 / 23.06** | 室内外分训 SOTA |
| OccAnyScene-DAv3 cross-scene | 67.96 / 59.51 | 35.83 / 22.87 | 联合训练仅 **-0.41 / -0.19** mIoU |
| OccAnyScene-DAv2 scene-specific | 64.98 / 55.42 | 33.86 / 20.46 | 效率默认骨干 |
| OccAnyScene-DAv2 cross-scene | 64.56 / 55.46 | 33.76 / 20.36 | 室内 mIoU 几乎持平 |
| SplatSSC† cross-scene | 57.42 / 46.80 | 31.24 / 17.86 | 直接跨场景适配掉 **~5.03 / 1.19** mIoU |

- **消融（DAv2 联合训练）：** 无 PFFA/FPGC 基线 47.21 / 17.39 mIoU；仅 FPGC 已大幅抬升；PFFA+FPGC 达 55.46 / 20.36。去掉深度残差伤害最大（室内 IoU -5.91）。
- **\(K\)：** 1→3 仅边际增益；遮挡补全主要靠 **邻像素交替深度增量**，而非同一视锥多层分离。
- **效率（Occ-ScanNet，RTX 4090）：** DAv2 **98.2 M / 86.4 ms / 670 MiB**，相对 EmbodiedOcc / SplatSSC 显存约 **-80.7% / -78.6%**；DAv3 时延仍约 **88.4 ms**。

### 4) 开源核查（步骤 2.5）
- **项目页：** 有 arXiv、**Code** 按钮、demo 视频、方法/结果叙事；Code 指向 [`RoboPerception/OccAnyScene`](https://github.com/RoboPerception/OccAnyScene)。
- **GitHub（2026-08-13）：** `main` 仅 `.gitignore` + `README.md` + `assets/`（teaser / framework / demo 媒体）。README 徽章写 **Code — release upon acceptance**；正文：「implementation, pretrained models, and training instructions will be released after paper acceptance。」无 LICENSE、无训练/推理脚本、无权重。
- **结论：** **部分开源（项目页 + 占位仓）/ 训练与推理待录用后发布** → wiki `## 源码运行时序图` 标不适用。勿把项目页 Code 按钮读成已可复现。

## 对 wiki 的映射

- 升格 [OccAnyScene 论文实体](../../wiki/entities/paper-occanyscene.md)
- 更新 [具身感知六种空间表征](../../wiki/concepts/embodied-perception-six-spatial-representations.md)、[2D→3D 语义提升 Gap](../../wiki/concepts/2d-to-3d-semantic-lifting-gap.md)、[导航·SLAM 栈](../../wiki/overview/navigation-slam-autonomy-stack.md)、[感知栈选型闭环](../../wiki/queries/robot-perception-stack-selection-loop.md)、[Humanoid Occupancy](../../wiki/entities/paper-notebook-humanoid-occupancy-enabling-a-generalized-multim.md)

## 当前提炼状态

- [x] 摘要 + PFFA/FPGC + 室内外表 + 开源边界
- [x] wiki 实体页与交叉引用
- [x] `sources/sites/` + `sources/repos/`（占位仓）
