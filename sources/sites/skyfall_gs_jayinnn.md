# skyfall_gs_jayinnn

> 来源归档（project site）

- **标题：** Skyfall-GS — Synthesizing Immersive 3D Urban Scenes from Satellite Imagery
- **类型：** site
- **原始链接：** <https://skyfall-gs.jayinnn.dev/>
- **镜像入口：** <https://skyfall-gs.jayinnn.dev/%E3%80%82>（同页，URL 含中文句号编码）
- **入库日期：** 2026-09-20
- **机构：** 国立阳明交通大学（NYCU）· UIUC · 萨拉戈萨大学 · UC Merced
- **论文：** [arXiv:2510.15869](https://arxiv.org/abs/2510.15869)
- **代码：** <https://github.com/jayin92/Skyfall-GS>
- **数据集：** [HF datasets](https://huggingface.co/datasets/jayinnn/Skyfall-GS-datasets) · [HF eval](https://huggingface.co/datasets/jayinnn/Skyfall-GS-eval) · [HF PLY](https://huggingface.co/jayinnn/Skyfall-GS-ply) · [Google Drive](https://drive.google.com/drive/folders/1Uugwpf7n5fj7k4UJRBuKUyrmkYcDRScQ?usp=sharing)
- **一句话说明：** Skyfall-GS 官方项目页：卫星 → 3DGS 城市街区合成；含 Abstract、两阶段 Method 图、**12 场景交互 Web 3DGS Viewer**（WASD 飞行）。

## 页面要点（2026-09-20 核查）

- **Abstract：** 卫星粗几何 + 扩散近景外观；课程式 IDU 迭代精炼；跨视角一致几何与更真实纹理。
- **Method：** (a) Reconstruction：3DGS + pseudo depth + appearance；(b) Synthesis：IDU + T2I prompt-to-prompt；(c) FlowEdit 示意。
- **Interactive Viewer：** JAX_004/068/214/260/164/168/175/264、NYC_004/010/219/336 等场景按钮；鼠标 + WASD 漫游。
- **Footer 链接：** GitHub `jayin92/skyfall-gs` 与 `Skyfall-GS`（同仓）；无独立 Hugging Face 页外链于 hero，但 README 列全 HF 资源。

## 开源结论

- **已开源** — 项目页 Code 链到 GitHub；训练/数据/PLY/viewer 均可获取；许可 Apache 2.0（以仓库 LICENSE 为准）。

## 对 wiki 的映射

- [paper-skyfall-gs](../../wiki/entities/paper-skyfall-gs.md)
- [sources/papers/skyfall_gs_arxiv_2510_15869.md](../papers/skyfall_gs_arxiv_2510_15869.md)
