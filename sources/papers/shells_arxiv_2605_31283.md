# shells_arxiv_2605_31283

> 来源归档（ingest）

- **标题：** Topologically Consistent Multi-view 3D Head Reconstruction via Coarse-Guided Layered Surface Sampling（SHELLS）
- **类型：** paper
- **来源：** arXiv:2605.31283（2026-05-29）；SIGGRAPH Conference Papers 2026（DOI: 10.1145/3799902.3811201）
- **作者：** Timo Bolkart, Daoye Wang, Prashanth Chandran（Google Switzerland）
- **入库日期：** 2026-07-30
- **最后更新：** 2026-07-30
- **一句话说明：** Google SHELLS：分层表面感知采样 + XCiT 前馈重建固定拓扑 ~18k 顶点人头；0.08 s、相对体积方法约 3.5× 加速与 88% 推理显存下降；仅合成数据训练可泛化到真实多视角采集。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §1）

- **链接：** <https://arxiv.org/abs/2605.31283>
- **核心贡献：** 提出 **SHELLS**（Semantic Head Estimation via Layered Local Sampling）：从标定多视角图像前馈预测 **稠密语义对应** 的固定拓扑人头网格。现有 ToFu / TEMPEH / GRAPE 等依赖体素/局部特征体积，采样代价与输出顶点数耦合，难以扩展到 ≥10k 顶点且易产生表面噪声。SHELLS 用稀疏全局采样图 → 粗网格 → 沿法向分层采样壳，将特征采样与最终分辨率解耦；共享 XCiT transformer **整体**回归顶点（注意力加权采样坐标和），相对体积基线推理显存约 **2.4 GB vs ~20 GB（−88%）**、速度约 **0.08 s vs 0.29 s（3.5×）**，中位配准误差降 **21%–29%**。训练 **仅用合成数据** 即可泛化到真实多视角棚拍。
- **对 wiki 的映射：**
  - [SHELLS 论文实体](../../wiki/entities/paper-shells-layered-surface-sampling.md)
  - [Face Anything](../../wiki/entities/paper-face-anything-4d-face-reconstruction.md)（面部 4D / 单目序列对照）
  - [GNM Head](../../wiki/entities/gnm-head.md)（参数化 3DMM 下游/对照）

### 2) 两阶段分层采样（§3.1–3.2）

- **链接：** <https://arxiv.org/html/2605.31283>
- **核心贡献：**
  - **特征：** 共享 DINOv2-B + LoRA（r=5）提多视角特征图；投影采样后做 mean–variance 融合（粗阶段）或 TEMPEH 式可见性加权融合（壳阶段）。
  - **Graph 粗阶段：** 16 层同心正二十面体稀疏图（\(n_g=2592\)）+ 下采样模板（\(n_c=3000\)）→ 粗网格 \(\hat V_c\)。
  - **Shell 精阶段：** 粗网格沿法向 \(\pm 4\,\mathrm{mm}\) 堆叠壳（\(3n_c=9000\) 点）→ 全分辨率模板（\(n_v=17821\)）注意力加权回归最终顶点。两阶段合计约 **11.6k** 采样点，远少于 ToFu/TEMPEH 每顶点 \(8^3\) 局部体积。
  - **预测形式：** \(\hat V = \mathrm{Softmax}(QK^\top/\sqrt{d_m})\,S\)（采样坐标的注意力加权和），非逐点独立偏移。
- **对 wiki 的映射：**
  - [SHELLS 论文实体](../../wiki/entities/paper-shells-layered-surface-sampling.md)
  - [视觉表征作为策略输入](../../wiki/concepts/visual-representation-for-policy.md)

### 3) 损失、数据与实现（§3.3–§4）

- **核心贡献：** V2V + V2P（法向分量）联合监督；按区域加权（唇/睑 5.0、皮肤/眉/耳/鼻 3.0）。合成数据：Wood et al. 流程 + 内部注册网格（≥2500 身份）→ Blender Cycles 13 视角、300k 对 / 2064 身份；训练 80/10/10 身份不交。H100 单卡约 2 周、900k step；前 500k 只训粗阶段+LoRA。训练期随机 **8–13** 视角 dropout，推理可少至 **2 视角**。
- **对 wiki 的映射：**
  - [humanoid-training-data-pipeline](../../wiki/queries/humanoid-training-data-pipeline.md)（合成面部监督 / 注册网格数据工厂）
  - [Sim2Real](../../wiki/concepts/sim2real.md)（合成→真实多视角泛化）

### 4) 实验：合成 / 真实对比与消融（§5）

- **基线：** 同合成数据重训 TEMPEH；3DMM 多视角参数平均；多视角 3DMM fitting（~35 s）。
- **合成（Table 1，face V2V median）：** SHELLS Final **1.22 mm** vs TEMPEH Refinement **1.71 mm**（约 −29%）。
- **真实采集（Table 2）：** V2V median **1.50 mm** vs TEMPEH **1.90 mm**（约 −21%）；P2S 均值略优、中位 TEMPEH 更贴扫描（局部拉点 vs 全局语义对应权衡）。网格质量：三角形形变 **0.38 vs 0.55**，翻转率约减半。
- **消融：** DINOv2+LoRA、壳阶段、采样图细分密度均关键；仅粗阶段缺中频细节。
- **对 wiki 的映射：**
  - [SHELLS 论文实体](../../wiki/entities/paper-shells-layered-surface-sampling.md)
  - [Face Anything](../../wiki/entities/paper-face-anything-4d-face-reconstruction.md)

### 5) 应用、遮挡与局限（§6）

- **应用：** 刚性对齐后可快速建 3DMM；逐帧 performance capture 无需时序滤波即可较平滑；全局注意力可补全发丝/衣物遮挡与口腔内侧拓扑一致顶点。
- **局限：** 极端舌姿因合成数据不足失败；18k 网格无细皱纹/毛孔（需另加位移/纹理网）；优化皮肤表面而非发须外包络；单视角病态。
- **开源核查（步骤 2.5，2026-07-30）：** 项目页 <https://syntec-research.github.io/SHELLS/> 仅提供 arXiv / PDF / BibTeX，**未列 GitHub / HF 代码或权重**；论文亦未给出代码 URL → 记为 **未开源（截至入库日）**。
- **对 wiki 的映射：**
  - [GNM Head](../../wiki/entities/gnm-head.md)（SHELLS 注册网格 → 建 3DMM 的上游）
  - [遥操作](../../wiki/tasks/teleoperation.md)（面部 performance / telepresence 上游）

## 关键数字速查

| 指标 | SHELLS | 体积基线（TEMPEH 等） |
|------|--------|----------------------|
| 输出拓扑 | ~18k 顶点固定拓扑 | TEMPEH 原设定偏更低分辨率体积细化 |
| 推理时间 | **0.08 s** | ~0.29 s（约 3.5×） |
| 推理显存 | **~2.4 GB** | ~20 GB（约 −88%） |
| 训练显存 | ~20 GB | ~65 GB（约 −70%） |
| 合成 V2V median（face） | **1.22 mm** | 1.71 mm |
| 真实 V2V median | **1.50 mm** | 1.90 mm |
| 最少视角 | **2**（训练随机 dropout） | 体积方法通常需更多视角 |

## 其他公开资料

- **项目页：** <https://syntec-research.github.io/SHELLS/> — 归档见 [sources/sites/shells-project.md](../sites/shells-project.md)
- **PDF：** <https://syntec-research.github.io/SHELLS/files/paper.pdf>
- **arXiv HTML：** <https://arxiv.org/html/2605.31283>

## 当前提炼状态

- [x] 摘要与核心方法摘录（≥5 条）
- [x] wiki 页面映射
- [x] 项目页源码开放核查（未开源）
