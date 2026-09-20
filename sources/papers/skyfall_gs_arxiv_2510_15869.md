# skyfall_gs_arxiv_2510_15869

> 来源归档（ingest）

- **标题：** Skyfall-GS: Synthesizing Immersive 3D Urban Scenes from Satellite Imagery
- **短名：** Skyfall-GS
- **类型：** paper
- **来源：** arXiv abs / PDF
- **原始链接：**
  - <https://arxiv.org/abs/2510.15869>
  - <https://arxiv.org/pdf/2510.15869>
- **项目页：** <https://skyfall-gs.jayinnn.dev/> — 归档见 [`sources/sites/skyfall_gs_jayinnn.md`](../sites/skyfall_gs_jayinnn.md)
- **代码：** <https://github.com/jayin92/Skyfall-GS> — [`sources/repos/skyfall_gs_jayinnn92.md`](../repos/skyfall_gs_jayinnn92.md)
- **数据集：**
  - <https://huggingface.co/datasets/jayinnn/Skyfall-GS-datasets>（JAX / NYC 训练集）
  - <https://huggingface.co/datasets/jayinnn/Skyfall-GS-eval>（评测视频与基线）
  - <https://huggingface.co/jayinnn/Skyfall-GS-ply>（预融合 PLY）
- **作者：** Jie-Ying Lee, Yi-Ruei Liu, Shr-Ruei Tsai, Wei-Cheng Chang, Chung-Ho Wu, Jiewen Chan, Zhenjun Zhao, Chieh Hubert Lin, Yu-Lun Liu
- **机构：** 国立阳明交通大学（NYCU）· UIUC · 萨拉戈萨大学 · UC Merced
- **版本：** arXiv:2510.15869；**ECCV 2026**（README 标注）
- **入库日期：** 2026-09-20
- **一句话说明：** 仅用多视角卫星影像合成 **城市街区尺度、可自由飞行漫游** 的 3DGS 城市场景：Stage 1 伪相机深度监督 + 多日期外观建模重建粗几何；Stage 2 **课程式 Iterative Dataset Update（IDU）** 用 T2I 扩散（FlowEdit / prompt-to-prompt）迭代精炼纹理与几何；官方代码、数据与融合 PLY **已开源**（Apache 2.0）。

## 核心摘录

### 1) 问题与动机
- 大规模可探索、几何准确的 **3D 城市** 对沉浸式/具身应用有价值，但缺少可训练泛化生成模型的大规模高质量 **真实 3D 扫描**。
- 替代路线：**现成卫星影像** 提供真实粗几何 + **开放域扩散模型** 合成近景高质外观，**无需昂贵 3D 标注**。
- 目标：**city-block 尺度**、**实时沉浸式 3D 探索**、跨视角一致几何与更真实纹理。

### 2) 方法要点（两阶段）

**Stage 1 — Reconstruction**
- 从多视角卫星图用 **3DGS** 重建初始场景。
- **伪相机深度监督**（pseudo-camera depth）缓解卫星 **视差有限**。
- **外观建模**（appearance modeling）处理 **多日期卫星图** 的光照变化。

**Stage 2 — Synthesis（IDU）**
- **Curriculum-driven Iterative Dataset Update（IDU）**：迭代用精炼渲染更新训练集。
- 预训练 **T2I 扩散** + **prompt-to-prompt editing**（实现侧 `--idu_use_flow_edit`，基于 FlowEdit）。
- 逐步提升几何完整性与 photorealistic 纹理。

### 3) 数据与场景
- 官方 **JAX** 与 **NYC** 数据集（HF / Google Drive）；项目页提供 **12+ 交互 3DGS 场景**（Residential、Office、Stadium、Union Square 等）。
- 自定义数据：COLMAP 或卫星管线（[`SatelliteSfM`](https://github.com/jayin92/SatelliteSfM)）→ `images/`、`transforms_*.json`、`points3D.txt` 等。

### 4) 评测（README / eval.py）
- 指标：**PSNR、SSIM、LPIPS、CLIP-FID、CMMD**。
- JAX 对比：mip-splatting、sat-nerf、eogs、corgs、ours_stage1/2 等；NYC：citydreamer、gaussiancity、corgs 等。

### 5) 开源核查（步骤 2.5，2026-09-20）
- **项目页：** Paper（arXiv）、**Code**（GitHub）、交互 Viewer、多场景按钮均已列出。
- **GitHub [`jayin92/Skyfall-GS`](https://github.com/jayin92/Skyfall-GS)：** **已开源** — `train.py` 两阶段、`eval.py`、`render_video.py`、`create_fused_ply.py`；`scripts/run_jax*.py` / `run_nyc*.py` 自动化；子模块 diff-gaussian-rasterization-depth 等。
- **数据/模型：** HF 数据集、评测包、**预融合 PLY** 可下载；Google Drive 镜像。
- **许可：** **Apache 2.0**。
- **结论：** **已开源**（训练/评测/渲染/可视化全流程可跑）。

## 对 wiki 的映射

- 升格 [Skyfall-GS 论文实体](../../wiki/entities/paper-skyfall-gs.md)
- 交叉 [Real2Sim 纵深](../../roadmap/depth-real2sim.md) Stage 1（卫星/航拍 → 可漫游 3DGS 资产）、[GS-Playground](../../wiki/entities/gs-playground.md)、[Spark 3DGS 渲染器](../../wiki/entities/spark-3dgs-renderer.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)、[PanoLOG/G²PS](../../wiki/entities/paper-panolog-ggps.md)（户外大场景 3DGS 对照）

## 当前提炼状态

- [x] 摘要 + 两阶段方法 + 开源边界
- [x] wiki 实体页与交叉引用
- [x] `sources/sites/` + `sources/repos/`
