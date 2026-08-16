# lego_leveled_language_gs_arxiv_2608_10057

> 来源归档（ingest）

- **标题：** LEGO: Leveled Language Gaussian Splatting
- **短名：** LEGO
- **类型：** paper
- **来源：** arXiv abs / HTML / PDF
- **原始链接：**
  - <https://arxiv.org/abs/2608.10057>
  - <https://arxiv.org/html/2608.10057v1>
  - <https://arxiv.org/pdf/2608.10057>
- **项目页：** <https://pz0826.github.io/LEGO-Webpage/> — 归档见 [`sources/sites/pz0826-lego-webpage.md`](../sites/pz0826-lego-webpage.md)
- **代码：** <https://github.com/WHU-USI3DV/LEGO> — 归档见 [`sources/repos/lego.md`](../repos/lego.md)
- **作者：** Yuning Peng<sup>1</sup>, Haiping Wang<sup>1,2</sup>, Yuan Liu<sup>2</sup>, Yipeng Lu<sup>1</sup>, Zhen Dong<sup>1</sup>, Bisheng Yang<sup>1</sup>
- **机构：** 武汉大学（WHU）；香港科技大学（HKUST）
- **版本：** arXiv:2608.10057（2026-08）；ECCV 2026
- **入库日期：** 2026-08-16
- **一句话说明：** 把多视角 SAM 掩码按共视与 3D 尺度重分级成与视距、绝对尺寸无关的结构层级，再蒸馏到解耦的 3DGS 特征场，并用 CLIP + 层级场景图做开放词汇分割与 LLM Chain-of-Retrieval。官方仓 **已开源、可运行**（`lego run` / `eval` / `viewer`）。

## 核心摘录

### 1) 问题
- 真实语义是层级的（花盆 → 花束 → 花蕾 → 花瓣），开放词汇 3D 理解不能只识别扁平概念。
- **粒度蒸馏（LangSplat / HiL-Splat / OccamLGS 等）**：直接把 SAM 的 whole / part / subpart 当 3D 层级。SAM 标签由 2D 相对粒度决定，近景把花切成花瓣、远景同一粒度只出花蕾 → 3D 层级模糊。
- **尺度蒸馏（LERF / GARField / SAGA）**：用绝对物理尺度切段。同类物体尺寸差大时，同一尺度参数会把大花切成花瓣、小花保持完整 → **语义–尺度脱钩**，要逐实例手调。

### 2) 方法要点
1. **几何初始化：** MASt3R-SfM 出位姿与点，再初始化 3DGS 场 \(\mathcal{P}\)（实现侧默认位姿用 COLMAP，可用 `gaussian.init=mast3r`）。
2. **掩码提升与尺度：** 像素映射到 3D 点后，掩码物理尺度 \(s_i=2\sqrt{\sum_d \mathrm{std}(\mathcal{P}_{m_i,d})^2}\)。
3. **局部峰值定级：** 对共视邻域做尺度直方图峰检测，得到粗到细峰 \(\mathcal{K}_i\)；\(l_i=\arg\min_l |s_i-p_l|\)。结构层级对视距与绝对尺寸不变。
4. **稠密层级指示器 \(\mathbb{I}_k(i,j)\)：** 单层重分级掩码稀疏。用单调性、递归包含、结构继承三条公理，把标签沿层级传播成像素对是否同实体。
5. **解耦特征：** 每个高斯 \(\mathbf{f}\in\mathbb{R}^{L\times d}\)，第 \(k\) 行只训第 \(k\) 层。对比损失 + 面积反比权重 + 正对 \(L_2\) + 3D/2D 单位球正则。论文默认 \(L=8,d=8\)。
6. **自顶向下聚类：** 父节点用 HDBSCAN 按子层特征切子簇，得到嵌套树。
7. **最优视角 CLIP（OVS）：** 可见度 × 2D 覆盖 × 与最佳 SAM 的 IoU，取 top-\(\tau\) 视角裁剪提 CLIP，避免多视角平均被遮挡污染。
8. **层级语言场景图：** 层级边（部分–整体）+ 邻接边（包围球相交）；LLM 把复合查询拆成 coarse-to-fine 链，做关系约束 beam search（CoR）。

### 3) 实验（论文报告摘要）
- **可提示分割：** NVOS mIoU **94.2** / mAcc **98.7**（相对 SAGA +1.6 mIoU）；SPIn-NeRF mIoU **94.2** / mAcc **99.3**。
- **开放词汇：** LERF-OVS 定位 mAcc **88.4**、分割 mIoU **68.4**（相对最强基线约 +4.1 / +4.4）；Mip-NeRF 360 mAcc **92.6**、mIoU **73.0**（约 +3.9 / +3.6）。Ramen 细粒度（玉米、洋葱片）相对最强基线约 +11.2 mAcc / +11.9 mIoU。
- **3D-OVS（附录）：** overall mIoU **96.5**。
- **CoR 120 条复合查询：** LEGO **51.6** mIoU；去 CoR 仅 **22.7**；LaGa+CoR **14.3**；BBQ / THGS / 原版 LaGa 均 <10。
- **消融（Mip-NeRF 360 Room）：** 去掉面积重加权 mAcc −6.9；再去 \(L_{pos}\) mIoU −1.3；再去特征归一化再 −1.6。
- **训练代价：** RGB 场 30k iter 后冻几何，再训 identity 10k；单卡 RTX 4090 约 **20–60 min/场景**，树 + CLIP 再 **5–10 min**。

### 4) 局限
- 离线、按场景优化；不是机载在线语义 SLAM。
- 依赖多视角 RGB + 位姿/SfM；动态、透明、强反光未展开。
- 单词/短语查询只扫前三层 CLIP，不用图；复合查询才走 LLM CoR。
- 官方仓 **CC BY-NC-SA 4.0**，无现成场景权重，需自训。

### 5) 开源核查（步骤 2.5）
- **项目页（2026-08-16）：** 页首 / 资源区 **Code** 指向 [`WHU-USI3DV/LEGO`](https://github.com/WHU-USI3DV/LEGO)；另有 arXiv 与实验室 [`WHU-USI3DV`](https://github.com/WHU-USI3DV/)。
- **仓库：** 完整 `src/lego` 管线、`lego` CLI、`benchmarks/` 评测、`configs/scenes/` 场景配置、`scripts/setup_env.sh` / `download_models.sh`。`checkpoints/` 仅 `.gitkeep`。
- **结论：** **已开源、可运行训练 / 评测 / 可视化**。wiki 须写 `## 源码运行时序图`。勿与斯坦福 [LEGS](../../wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md) 或 LEGO-SLAM 混名。

## 对 wiki 的映射

- 升格 [LEGO 论文实体](../../wiki/entities/paper-lego-leveled-language-gaussian-splatting.md)
- 更新 [2D→3D 语义提升 Gap](../../wiki/concepts/2d-to-3d-semantic-lifting-gap.md)、[感知栈选型闭环](../../wiki/queries/robot-perception-stack-selection-loop.md)、[SAM](../../wiki/entities/paper-segment-anything.md)、[OV-SAM3D](../../wiki/entities/ov-sam3d.md)

## 当前提炼状态

- [x] 摘要 + 层级 vs 粒度/尺度 + 评测表 + 开源边界
- [x] wiki 实体页、仓库归档与交叉引用
- [x] `sources/sites/` + `sources/repos/`
