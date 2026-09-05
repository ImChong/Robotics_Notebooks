# World Labs Blog：Marble — A Multimodal World Model

> 来源归档（ingest）

- **标题：** Marble: A Multimodal World Model
- **类型：** blog
- **URL：** <https://www.worldlabs.ai/blog/marble-world-model>
- **机构：** World Labs
- **日期：** 2025-11-12（文内标注；GA 发布日）
- **入库日期：** 2026-09-05
- **产品：** <https://marble.worldlabs.ai/>
- **文档：** <https://docs.worldlabs.ai/>
- **一句话说明：** Marble 作为 **多模态生成式世界模型** 向全员开放：文本 / 图 / 多图 / 视频 / 粗 3D（Chisel）生成可编辑、可扩展、可组合的持久 3D 世界，导出 **高斯 splat / mesh / 视频**；定位空间智能产品，不是具身像素 WM。

## 核心摘录

### 1) 产品定位（文首）

- 世界模型应 **重建、生成、仿真** 3D 世界，并让人与 agent 交互。
- 两月前预览仅图/文；本日 **GA**，能力扩到多模态输入、交互编辑 / 扩展 / 组合，导出 splat、mesh、视频。
- 同期上线 **Marble Labs**（案例、教程、文档聚合）。入口：`marble.worldlabs.ai`。

### 2) 多模态输入与控制梯度

| 输入 | 控制程度 | 要点 |
|------|----------|------|
| 文本 / 单图 | 低 | 未写清的细节由模型发明；可接外部文生图再 lift 到 3D |
| 多图（前/后/左/右） | 中 | 分视角指定外观，Marble 缝合成统一世界；也可用真实现场多角度照片 |
| 视频 | 中 | 短视频提供更丰富空间信息 |
| **Chisel** 粗 3D + 文本 | 高 | **结构与风格解耦**：盒/面/导入资产定布局，prompt 定外观；同一布局可换风格 |

### 3) 编辑、扩展、组合

- **World editing：** 局部改物 / 换风格 / 改大结构（例：后墙改舞台、台面改黑花岗岩）。
- **Expand：** 指定区域一次扩展，修远场糊边、加大可漫游范围。
- **Compose：** 用户自控布局，把多个世界拼成更大空间（文内火车场景）。

### 4) 导出与视频增强

- **最高保真：** 高斯 splat；浏览器用开源 **Spark**（THREE.js）渲染。
- **Mesh：** **collider**（粗物理）与 **高保真三角网**（尽量贴近 splat）。
- **视频：** 像素级相机控制；**Enhanced video** 去伪影并加烟/火/流水等动态，仍贴 3D 结构。

### 5) 开源边界（步骤 2.5，2026-09-05）

- 博客 **未列** Marble 权重 / 训练代码 GitHub。
- **已开源（渲染与示例）：** Spark（[sparkjs.dev](https://sparkjs.dev/)）；文档另列 Interactive World Examples（Rapier + Spark 等）。
- **产品 / API：** 闭源 SaaS + [World API](https://docs.worldlabs.ai/api/index.md)（`platform.worldlabs.ai`），需积分。
- 结论：**部分开源** — 生成模型未开放；Web 渲染与交互示例开放。

## 对 wiki 的映射

- [Marble 实体](../../wiki/entities/marble-world-model.md)
- [World Labs](../../wiki/entities/world-labs.md)
- [Spark](../../wiki/entities/spark-3dgs-renderer.md)
- [Atlas](../../wiki/entities/atlas-world-model.md) — 后续 omni 底座叙事
- [生成式世界模型](../../wiki/methods/generative-world-models.md)
