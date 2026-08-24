# sculpt_arxiv_2608_13541

> 来源归档（ingest）

- **标题：** SCULPT: Subtractive Composition for 3D Part Generation
- **类型：** paper
- **来源：** arXiv:2608.13541（2026）；项目页 <https://sculpt-part.github.io/>
- **作者：** Sikuang Li, Chen Yang, Jiemin Fang, Jiazhong Cen, Yuhe Wei, Jichen Pang, Wei Shen, Qi Tian（上海交通大学 × 华为；* 共一，✉ 通讯，§ 项目负责）
- **入库日期：** 2026-08-24
- **最后更新：** 2026-08-24
- **一句话说明：** SCULPT 把 part-aware 3D 生成表述为 **减法式组合**：在 TRELLIS.2 结构化潜空间上，联合 split predictor 每步同时去噪「提取部件 + 更新余量」，原生稀疏支撑并集上保留 interface shell；PartObjaverse 上 part/object 级 CD 与 F1@.05 SOTA，可泛化到 T2I 与真实照片。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §1）

- **链接：** <https://arxiv.org/abs/2608.13541>
- **核心贡献：** 现有 part-aware 3D 生成分两路：**分割式**（先生成整体再切分，边界与材质在生成后决定）与 **加法式**（预定义布局/槽位合成部件再拼合，易出现缝隙、穿插与材质不连续）。SCULPT 提出 **subtractive composition**：从完整对象的结构化 3D 潜表示出发，迭代应用 **联合 split predictor**，每步 **耦合去噪** 一个提取部件与更新余量；二者在 **原生稀疏支撑并集** 上处理，邻接支撑可重叠形成 **interface shell**，无需焊接/对齐/重缩放。rollout 在余量支撑为空或达安全上限 \(K_{\max}=24\) 时终止，部件数随对象自适应。
- **对 wiki 的映射：**
  - [SCULPT 论文实体](../../wiki/entities/paper-sculpt-subtractive-3d-part-generation.md)
  - [PhysForge](../../wiki/entities/paper-physforge-physics-grounded-3d-assets.md)（同为部件化 3D 资产生成对照）
  - [PhysX-Omni](../../wiki/entities/physx-omni.md)（TRELLIS 系整体生成上游）

### 2) 方法：结构化潜空间与联合 split predictor（§3）

- **链接：** <https://arxiv.org/html/2608.13541>
- **核心贡献：**
  - **骨干：** 预训练 **TRELLIS.2** 图像条件生成器 \(G_\phi\) 得完整对象潜变量 \(\boldsymbol{z}_0=(\boldsymbol{z}^v,\boldsymbol{z}^g,\boldsymbol{z}^m)\)（稀疏结构 / 几何 / 材质三阶段 O-Voxel 结构化潜空间）。
  - **联合 predictor \(P_\theta\)：** 三阶段 **decomposition flow transformer**；每阶段含 **joint denoising branch**（自 TRELLIS.2 初始化）与 **remainder-control branch**（ControlNet 式零初始化残差注入当前余量）。
  - **训练序列：** 从 PartVerse-XL 部件网格按质心 \(z\)–\(x\)–\(y\) 字典序构造 subtractive 监督 \((\mathcal{I},\boldsymbol{z}_{i-1};\hat{\boldsymbol{z}}_i,\boldsymbol{z}_i)\)；部件与余量 **独立编码** 允许接触区支撑重叠。
  - **稀疏阶段 composition loss \(\mathcal{L}_{\mathrm{comp}}\)：** 可微并集 \(\boldsymbol{u}_i\) 与输入余量支撑 BCE 对齐（\(\lambda_{\mathrm{comp}}=0.1\)），训练「覆盖当前对象」而非硬不相交划分。
  - **推理：** 阈值 \(\tau_{\mathrm{occ}}\) 硬化支撑并 **clip 到 \(\mathcal{O}_{i-1}\)**，保证非扩张 subtractive 状态；空余量或达 \(K_{\max}\) 停止，非空 capped remainder 作为额外输出保留。
- **对 wiki 的映射：**
  - [SCULPT 论文实体](../../wiki/entities/paper-sculpt-subtractive-3d-part-generation.md)
  - [DiffGI](../../wiki/entities/paper-diffgi.md)（另一路 3D 生成表征对照）

### 3) 训练数据与实现（§4.1）

- **训练：** **PartVerse-XL**（源自 Objaverse-XL，人工精修部件网格）；过滤后 **37,425** 对象、**330,455** 条 part–remainder split；与评测集 SHA-256 去重防泄漏。
- **初始化：** 三阶段 predictor 自公开 **TRELLIS.2** checkpoint 初始化，各 **30** transformer blocks；**32 GPU**、batch 1/GPU、AdamW lr \(10^{-4}\)、共 **800K** steps；几何/材质阶段丢弃 token 数 <8 或 >8192 的 split（约 1.5% split / 2.3% 资产）。
- **对 wiki 的映射：**
  - [SCULPT 论文实体](../../wiki/entities/paper-sculpt-subtractive-3d-part-generation.md)
  - [3DGenStudio](../../wiki/entities/3dgenstudio.md)（TRELLIS.2 工程生态）

### 4) 实验：PartObjaverse 与泛化（§4.2–4.4）

- **评测：** **PartObjaverse**（200 mesh，实例级 + 语义组 + 组装对象三级）；指标 CD ↓、F1@.1 / F1@.05 ↑；匈牙利匹配部件。
- **基线：** 直接部件生成 **Part123、OmniPart**；后验分解 **TRELLIS/TRELLIS.2 + PartField/SAM3D (+ HoloPart)**。
- **主要结果（Table 1）：**
  - **Part level CD：** SCULPT **0.0107** vs OmniPart 0.0136（−21%）、TRELLIS.2+PartField 0.0115（−7.0%）。
  - **Object level CD：** **0.0020** vs OmniPart 0.0032；F1@.05 **0.9212** vs 0.8732。
  - **消融：** 仅适配稀疏阶段 CD 0.0439；去 \(\mathcal{L}_{\mathrm{comp}}\) → 0.0279；去 clip → 0.0260。
- **定性：** 四张数据集图 + 一张 T2I + 一张真实照片；各部件带纹理，组装保持共享物体坐标系。
- **对 wiki 的映射：**
  - [SCULPT 论文实体](../../wiki/entities/paper-sculpt-subtractive-3d-part-generation.md)
  - [PhysForge](../../wiki/entities/paper-physforge-physics-grounded-3d-assets.md)（PartObjaverse 评测交叉）

### 5) 局限与开源核查（§4 / 项目页）

- **局限：** rollout 上限 24 步；极复杂对象可能 capped remainder；依赖 TRELLIS.2 结构化潜空间与 PartVerse-XL 监督；未强调关节/物理仿真字段（与 PhysForge / PhysX-Omni 不同赛道）。
- **开源核查（步骤 2.5，2026-08-24）：** 项目页 <https://sculpt-part.github.io/> 仅提供交互 demo、方法说明、BibTeX，**未列 GitHub / HF 代码或权重** → 记为 **未开源（截至入库日）**。
- **对 wiki 的映射：**
  - [Articraft](../../wiki/entities/articraft.md)（程序化 sim-ready 资产生成对照）
  - [SCULPT 项目页归档](../sites/sculpt-part-project.md)

## 关键数字速查

| 指标 | SCULPT | 最强对照（论文报告） |
|------|--------|----------------------|
| Part-level CD | **0.0107** | OmniPart 0.0136；TRELLIS.2+PartField 0.0115 |
| Part-level F1@.05 | **0.7599** | OmniPart 0.7025 |
| Object-level CD | **0.0020** | OmniPart 0.0032 |
| Object-level F1@.05 | **0.9212** | OmniPart 0.8732 |
| 训练 split 数 | 330,455 | 37,425 对象 |
| 最大 rollout | \(K_{\max}=24\) | 空余量提前终止 |
| vs TRELLIS.2+PartField part CD | **−7.0%** | 后验分解最强基线 |

## 其他公开资料

- **项目页：** <https://sculpt-part.github.io/> — 归档见 [sources/sites/sculpt-part-project.md](../sites/sculpt-part-project.md)
- **PDF：** <https://arxiv.org/pdf/2608.13541>
- **arXiv HTML：** <https://arxiv.org/html/2608.13541>

## 当前提炼状态

- [x] sources 归档完成
- [x] 项目页开源核查完成（未开源）
- [x] wiki 实体页已升格
