# Human Atlas 在线演示

> 来源归档

- **标题：** Human Atlas — Live Demo
- **类型：** site（Vercel 托管静态演示）
- **链接：** <https://human-atlas-seven.vercel.app>
- **源码：** <https://github.com/ashemag/human-atlas>
- **入库日期：** 2026-09-06
- **一句话说明：** BodyParts3D 4.0 参考解剖的 **交互 3D 浏览器**：系统层、exploded 拆解、3,432 概念搜索与 isolate 详情。
- **沉淀到 wiki：** [`wiki/entities/human-atlas.md`](../../wiki/entities/human-atlas.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **前端** | 对应 GitHub **MIT** 开源仓 |
| **解剖数据** | BodyParts3D **CC BY 4.0**；页内/仓库 `ATTRIBUTION.md` 保留署名 |
| **定位** | **教育探索器**，非临床诊断或手术规划工具 |

## 页面能力（与 README 对齐，2026-09-06）

- 3D 轨道上点选 **2,234** mesh；**15** 系统 preset
- Exploded view：可见结构 spaced inventory
- 搜索解剖名称与 source identifier
- Isolate 选中结构 + 详情侧栏；移动端布局已测 390×844 等
- 可选 WebMCP：兼容浏览器可暴露 anatomy search/inspect tools

## 数据说明

- **BodyParts3D 4.0** 成人男性参考；**不覆盖** 全部人类变异与结构
- Mesh（2,234）≠ FMA concept（3,432）；concept 可 group 多 mesh
- 历史版本曾含 HuBMAP 女性参考器官集；**当前 release 未打包**（见仓库 `ATTRIBUTION.md`）

## 对 wiki 的映射

- 实体：[`wiki/entities/human-atlas.md`](../../wiki/entities/human-atlas.md)
- 仓归档：[`sources/repos/ashemag_human_atlas.md`](../repos/ashemag_human_atlas.md)
