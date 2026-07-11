# PEAR 项目页（wujh2001.github.io/PEAR）

> 来源归档

- **标题：** PEAR — Pixel-aligned Expressive humAn mesh Recovery
- **类型：** site（项目主页）
- **链接：** <https://wujh2001.github.io/PEAR/>
- **论文：** <https://arxiv.org/abs/2601.22693>
- **代码：** <https://github.com/Pixel-Talk/PEAR>
- **入库日期：** 2026-07-11
- **一句话说明：** SIGGRAPH 2026 官方项目页：单图 EHM-s 恢复、与 SAM3D-Body / OSX / SMPLest / Multi-HMR 的定性对比视频，以及实时面部表情/手部/全身动捕与虚拟人驱动演示。

---

## 页面要点（ingest 快照）

| 模块 | 内容 |
|------|------|
| 定位 | 像素级对齐的表意人体网格恢复；单图 **<0.01s**，无部位裁剪 |
| 架构图 | 统一 ViT encoder–decoder；SMPL-X + FLAME 双头；头尺度 $s$ |
| 对比 | **vs SAM3D-Body**：上身像素对齐更准，推理约 **100×** 更快 |
| 脸/手/全身 | 相对 OSX、SMPLest、Multi-HMR 的 UBody / 动捕序列对比 |
| 下游 | 100 FPS 推理驱动 **50 FPS** 实时动画接口 |
| 极端场景 | 运动模糊、遮挡、强光照、宽松衣物与长发 |
| 资源 | Paper / Code 链接；Dataset **coming soon** |

## 对 wiki 的映射

- 主实体页：**`wiki/entities/paper-pear-pixel-aligned-expressive-hmr.md`**
- 论文摘录：**`sources/papers/pear_arxiv_2601_22693.md`**
