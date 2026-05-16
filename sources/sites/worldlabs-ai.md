# World Labs（worldlabs.ai）

> 来源归档（ingest 关联资料）

- **标题：** World Labs
- **类型：** site / company / research product
- **官方入口：** <https://www.worldlabs.ai/>
- **入库日期：** 2026-05-16
- **一句话说明：** 自称「空间智能（spatial intelligence）」公司与前沿 **世界模型** 研发方；公开叙事强调模型可 **感知、生成、推理并与 3D 世界交互**，覆盖叙事、设计到仿真类用例；首款面向创作者的产品为 **Marble**（浏览器内生成可漫游、可编辑的持久 3D 世界），并维护开源 **Spark** 3D Gaussian Splatting（3DGS）Web 渲染器（THREE.js + WebGL2）。
- **与本次 ingest 的关系：** 作为 [`wiki/entities/world-labs.md`](../../wiki/entities/world-labs.md) 的原始资料锚点；与仓库内 [生成式世界模型](../../wiki/methods/generative-world-models.md)、[GS-Playground](../../wiki/entities/gs-playground.md) 等主题相邻（三维世界表征与 3DGS 管线），但 **Marble 侧重生成式 3D 内容与交互编辑**，不等同于具身论文里常见的「像素视频世界模型」定义。

## 官方站点与产品入口（检索自 2026-05-16 公开页面）

| 资源 | URL | 备注 |
|------|-----|------|
| 首页 | <https://www.worldlabs.ai/> | 品牌叙事、Marble / Marble Labs / 博客入口 |
| About | <https://www.worldlabs.ai/about> | 团队与投资方概述；创始人公开表述为 Fei-Fei Li、Justin Johnson、Christoph Lassner、Ben Mildenhall |
| Marble（产品） | <https://marble.worldlabs.ai/> | 多模态输入生成 3D 世界、布局控制、编辑、导出等能力介绍 |
| Marble Labs | <https://www.worldlabs.ai/labs> | Showcase / Case studies / 教程与文档聚合 |
| Spark（开源 3DGS 渲染） | <https://sparkjs.dev/> | 官方文档与示例；博客说明与 Marble 同期研发 |

## 博客与深度稿（节选主题，便于 wiki 溯源）

| 文章 | URL | 技术要点（归纳） |
|------|-----|------------------|
| Spark 2.0：流式 3DGS 世界 | <https://www.worldlabs.ai/blog/spark-2.0> | Spark 2.0：面向超大 splat 场景的 **LoD splat 树**、**渐进式流式加载**、**.RAD** 随机访问格式、**虚拟显存式分页**；全局 splat 排序 + Web Worker 上 Wasm（Rust）遍历；与 THREE.js / WebGL2 生态对齐 |
| 3D as code | <https://www.worldlabs.ai/blog/3d-as-code> | 将 3D 视作人类与 AI 协同编辑、模拟与共享空间的通用媒介的叙事稿 |
| Funding 2026 | <https://www.worldlabs.ai/blog/funding-2026> | 2026 年融资与公司愿景更新（定量条款以官方披露为准） |

## 对 wiki 的映射

- [`wiki/entities/world-labs.md`](../../wiki/entities/world-labs.md) — 公司定位、Marble / Spark / Marble Labs 与和「生成式世界模型」「3DGS 仿真」知识节点的关系。
