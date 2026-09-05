# World Labs 文档（docs.worldlabs.ai）

> 来源归档（ingest）

- **标题：** Welcome to Marble / World Labs Documentation
- **类型：** site（产品文档）
- **发布方：** World Labs
- **原始链接：** <https://docs.worldlabs.ai/>
- **索引：** <https://docs.worldlabs.ai/llms.txt>
- **产品：** <https://marble.worldlabs.ai/>
- **GA 博客：** <https://www.worldlabs.ai/blog/marble-world-model>
- **入库日期：** 2026-09-05
- **一句话说明：** Marble 创作者文档 + **World API**：多模态生成、编辑/扩展/Studio、导出规格、模型档位与积分；机器人侧只把它当 **外观/漫游资产源**，不是物理仿真器。

## 文档要点（2026-09-05）

### 界面

Gallery / Create / Studio（Compose、Record）。桌面完整；移动端缺高级编辑、Chisel、pano 查看。

### 创建入口

Preset、文本、单图、多图（方向或 Auto Layout）、360° 全景、视频（&lt;100MB）、**Chisel** 粗 3D、Reuse prompt。

### 编辑

Pano edit（选区 + 自然语言）、Click and expand、Variation。

### 生成时间（文档估计）

| 操作 | 约时 |
|------|------|
| 文/图/3D → pano | ~30 s |
| 多图/视频 → pano | ~2 min |
| Draft（任意输入） | ~20 s |
| Create world | ~5 min |
| Expand world | ~5 min |
| Edit pano | ~20 s |
| High-quality mesh | ~1 hr |

### 模型档位（`/marble/models` + API `/api/models`）

| 产品名 | API `model` | 文档定价（credits / 次） |
|--------|-------------|--------------------------|
| Marble 1.1 Plus | `marble-1.1-plus` | 1500 底 + 0–1500 可变（自动扩空间） |
| Marble 1.1 | `marble-1.1` | 1500（推荐入门） |
| Marble 1.0 | `marble-1.0` | 1500（legacy；API 未指定时仍默认，将改 1.1） |
| Marble 1.0 Draft | `marble-1.0-draft` | 150（最快试 prompt） |

legacy API 别名：`Marble 0.1-plus` → `marble-1.0`，`Marble 0.1-mini` → `marble-1.0-draft`。

### 导出（付费门禁）

- **Free：** 能生成，**不能导出**。
- **Standard：** splat（SPZ / PLY）、360 pano、**collider mesh**。
- **Pro：** 高保真 textured **GLB** + 商用权利。

规格：SPZ/PLY 约 **2M** 或低分 **500k** splat；collider GLB **100–200k** 三角、约 3–4 MB；HQ mesh ~**600k** 贴图 GLB 或 ~**1M** 顶点色 GLB，约 100–200 MB，每用户约 **4 次/小时**。坐标系默认 **OpenCV**（+x 左、+y 下、+z 前）；进 OpenGL DCC 需 Y/Z 乘 −1。

文档写明：**splat 是最高保真**；collider **不要当视觉 mesh**；HQ mesh 在细杆、透明、天空、未覆盖区会有洞/floater。

### World API

`platform.worldlabs.ai`；文档含 media upload、worlds generate/export、pano depth→RGB、operations 轮询、credits、OpenAPI、agent skill。默认模型仍 `marble-1.0`（将改 `marble-1.1`）。

### 交互示例（开源示例，非模型权重）

Spark Physics（Three.js + Rapier + 碰撞 mesh）、Image Blaster、Collider Builder（`.spz`/`.rad` → `.glb`）、第三人称 / 第一人称 splat 控制器。

## 开源边界（步骤 2.5，2026-09-05）

| 组件 | 状态 |
|------|------|
| Marble 生成模型 / 权重 | **未开源**（SaaS + 付费 API） |
| Spark 渲染 | **已开源** — [sparkjs.dev](https://sparkjs.dev/) |
| Interactive World Examples | **示例开源**（物理/控制器模板） |
| 导出插件文档 | Unreal / Unity / Blender / Houdini 集成说明公开 |

## 对 wiki 的映射

- [Marble 实体](../../wiki/entities/marble-world-model.md)
- [World Labs](../../wiki/entities/world-labs.md)
- [Spark](../../wiki/entities/spark-3dgs-renderer.md)
- [GS-Playground](../../wiki/entities/gs-playground.md) — 对照：训练吞吐 vs 创作者导出
