# MeRoPE Project Page

> 来源归档

- **标题：** MeRoPE: Metric Rotary Position Embedding for Camera-Controlled Video Generation
- **类型：** site / project page
- **URL：** <https://qiaozhijian.github.io/merope/>
- **论文：** <https://arxiv.org/abs/2609.01252>
- **机构：** 香港科技大学（HKUST）；卓驭科技（Zhuoyu Technology）
- **入库日期：** 2026-09-06
- **一句话说明：** 官方项目页：射线几何 token 标注、范数保持 attention 公式、nuScenes / PanShot / Real-to-Sim 视频 demo 与 BibTeX。

## 开源核查（2026-09-06）

| 入口 | 状态 |
|------|------|
| Code | **未列链接** — 页头/Footer/正文无 GitHub、Hugging Face、Zenodo |
| 论文 | 摘要写 "Code will be made publicly available" |
| Demo | 嵌入式 nuScenes、PanShot、Real-to-Sim 视频与 attention 可视化 |
| arXiv | [2609.01252](https://arxiv.org/abs/2609.01252) |

**结论：** **宣称将开源 / 待发布** — 以项目页实际链接为准，截至入库日无法复现训练与推理。

## 页面内容要点

- 每个 video token 携带时间 + 校准相机射线（原点与方向），无需 VGGT / Depth Anything 等 3D 重建前处理
- 几何进入 attention：$\mathbf{q}_a^\top\mathcal{U}_{ab}\mathbf{k}_b$ 替代标准点积；对比 UCPE 齐次块与 MeRoPE 四块对角结构
- nuScenes（Wan2.2 TI2V-5B）相机路径控制；PanShot（Wan2.1 T2V-1.3B）跨 FoV 泛化
- Real-to-Sim：检索图像 + History SA 长 rollout（含跨天气）

## 对 wiki 的映射

- 论文摘录：[`sources/papers/merope_arxiv_2609_01252.md`](../papers/merope_arxiv_2609_01252.md)
- 沉淀 **[`wiki/entities/paper-merope.md`](../../wiki/entities/paper-merope.md)**
