# KeenTools FaceBuilder for Blender（产品页）

- **类型**：网站 / 产品主页
- **入口**：<https://keentools.io/products/facebuilder-for-blender>
- **主体**：KeenTools
- **收录日期**：2026-09-24
- **抓取说明**：以 **2026-09-24** 对首页公开文案、定价 FAQ 与系统要求的抓取为准；订阅价格与版本号会随商业策略更新。
- **代码（add-on 适配层）**：<https://github.com/KeenTools/keentools-blender>（GPLv3，开源）
- **核心库**：KeenTools Core Library — **非独立开源**；随订阅分发，见 [keentools-blender.md](../repos/keentools-blender.md)

## 一句话

**FaceBuilder for Blender** 是 KeenTools 的 **Blender 插件**：用 **少量照片** 在 Blender 内构建 **写实 3D 头部**（含自动对齐、实时雕刻预览、多视角纹理融合），并支持 **ARKit 兼容 FACS blendshape**、**FaceTracker 面部表演**、**MetaHuman / Character Creator 4 导出** 与 **GeoTracker 实拍合成** 等下游集成。

## 为什么值得保留

- 机器人知识库中 **数字人化身、遥操作界面、人类参考外观** 常与 **照片→3D 头部→引擎角色** 管线交叉，但此前缺少对 FaceBuilder 这一主流 **单/少图头部重建** 工具的独立溯源页。
- 与 [MetaHuman](./metahuman-epic-docs.md) 的 **Mesh to MetaHuman** 工作流有直接官方集成（见 [keentools-fbb-metahuman.md](./keentools-fbb-metahuman.md)），是 **Blender DCC → UE 数字人** 链路上的关键中间件。
- 项目页 FAQ 明确 **add-on 开源 / Core Library 闭源** 边界，避免 wiki 误写「完全开源可复现算法」。

## 开源状态核查（2026-09-24，步骤 2.5）

| 组件 | 开放程度 | 说明 |
|------|----------|------|
| **Blender add-on 源码** | **已开源** | FAQ 指向 GitHub；仓库 [KeenTools/keentools-blender](https://github.com/KeenTools/keentools-blender)，**GPLv3** |
| **KeenTools Core Library** | **闭源 / 订阅** | 含网格对齐、重建等核心实现；联网环境可由 add-on 自动下载；不可脱离 KeenTools 插件独立商用 |
| **Cloud API** | **商业 API** | 项目页 Footer 链至 Cloud API；与本地 Core 许可分离 |

> 结论：**部分开源** — UI 适配层与安装逻辑可审计，**几何重建与对齐算法在 Core Library 内闭源**。

## 公开产品能力（编译自产品页）

| 能力 | 摘要 |
|------|------|
| **少图/单图头部** | 4–8 张（或非中性表情照片）构建写实头部；支持单图近似 |
| **Auto align** | AI 辅助网格与照片对齐 |
| **实时雕刻** | 拖拽参考点匹配图像，即时 3D 反馈 |
| **One-click texturing** | 多视角纹理融合 |
| **LOD** | 高/中/低模切换与导出 |
| **FACS blendshape** | 内置 **51** 个 **ARKit 兼容** FACS 形变；支持 **Live Link Face** |
| **FaceTracker** | 同生态插件：参考视频 + 匹配几何捕捉面部表演（不离 Blender） |
| **MetaHuman 导出** | 兼容 MetaHuman UV 的 **MH texture**；配合 Mesh to MetaHuman（见集成页） |
| **Character Creator 4** | 一键导出 CC4 |
| **GeoTracker** | 实拍镜头 CGI 合成（同 KeenTools 生态） |
| **管线出口** | 导出至任意支持 3D 格式的下游工具 |

## 许可与定价（产品页摘要）

- **15 天免费试用**（首次启动 Core Library，全功能无限制）。
- **订阅制**：面向 Core Library；FaceBuilder add-on 本身 FAQ 称不单独订阅，但 **依赖 Core**。
- **定价档（2026-09-24 页内）**：FaceBuilder 单插件约 **$15.99/月** 或 **$699/年**（node-locked）；与 FaceTracker / GeoTracker 组合有更高档位。
- **Nuke/AE/Houdini 许可不覆盖 Blender** — 各平台独立授权。
- **取消订阅后**：可打开工程并导出结果，但 **不能再编辑 FaceBuilder 模型**。
- **模型商用**：FAQ 明确用户拥有插件创建模型的所有权，可出售。

## 系统要求

- 官方 **64 位 Blender 2.80+**（须从 [blender.org](https://www.blender.org/) 下载；Linux 发行版打包版常不兼容）。
- **64 位** Windows / Linux / macOS（Intel 或 Apple Silicon）。
- 需要能运行 Blender **3D 视口** 的 GPU；无 GPU 时可能出现照片视图黑屏。

## 对 wiki 的映射

- 主实体页：[wiki/entities/keentools-facebuilder.md](../../wiki/entities/keentools-facebuilder.md)
- MetaHuman 集成：[keentools-fbb-metahuman.md](./keentools-fbb-metahuman.md)
- 代码归档：[keentools-blender.md](../repos/keentools-blender.md)
- 交叉引用：[wiki/entities/blender.md](../../wiki/entities/blender.md)、[wiki/entities/metahuman.md](../../wiki/entities/metahuman.md)

## 参考链接

- 产品页：<https://keentools.io/products/facebuilder-for-blender>
- 下载：<https://keentools.io/download/facebuilder-for-blender>
- GitHub（add-on）：<https://github.com/KeenTools/keentools-blender>
- Core Library 下载：<https://keentools.io/download/core>
