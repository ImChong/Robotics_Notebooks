# EJShim/diffgi

> 来源归档

- **标题：** DiffGI（官方仓 / 项目页镜像）
- **类型：** repo
- **组织 / 作者：** Eungjune Shim 等（CLO Virtual Fashion）
- **代码：** <https://github.com/EJShim/diffgi>
- **论文：** <https://arxiv.org/abs/2607.13365>
- **项目页：** <https://ejshim.github.io/diffgi/>
- **许可：** 截至入库日根目录未见 LICENSE / 可运行源码
- **入库日期：** 2026-07-27
- **一句话说明：** 公开 GitHub 名与项目相关，但当前仅含 `docs/`（GitHub Pages 静态站）与 `.gitignore`；**不是**可复现训练/推理实现。

## 开源核查（2026-07-27）

| 项 | 状态 |
|----|------|
| 仓库可见 | 是（公开） |
| 树内容 | `.gitignore`、`docs/`（`index.html`、静态图/视频、demo） |
| README / train / eval / ckpt | **无** |
| 项目页 Code | **Code (soon)** |
| 结论 | **待发布** — 有空壳/站点仓，无可运行官方实现 |

## 仓库入口（当前）

| 路径 | 作用 |
|------|------|
| `docs/index.html` | 项目页正文 |
| `docs/static/images/*` | teaser / method / results 图 |
| `docs/static/videos/teaser.mp4` | 演示视频 |

> 一旦作者释出训练/推理代码，应更新本页「开源核查」并补 `wiki/entities/paper-diffgi.md` 的「源码运行时序图」。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [DiffGI](../../wiki/entities/paper-diffgi.md) | 论文实体：TSDF GI + DMS + 潜扩散 |
| [ClothTransformer](../../wiki/entities/paper-clothtransformer-unified-latent-cloth-simulation.md) | 薄壳服装网格下游仿真对照 |
| [PhysForge](../../wiki/entities/paper-physforge-physics-grounded-3d-assets.md) / [Articraft](../../wiki/entities/articraft.md) | sim-ready / 资产生成谱系对照 |

## 参考来源

- 项目页：<https://ejshim.github.io/diffgi/>
- 论文：<https://arxiv.org/abs/2607.13365>
