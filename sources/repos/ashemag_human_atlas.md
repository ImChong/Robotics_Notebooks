# ashemag/human-atlas

> 来源归档

- **标题：** Human Atlas
- **类型：** repo
- **组织：** ashemag
- **代码：** <https://github.com/ashemag/human-atlas>
- **在线演示：** <https://human-atlas-seven.vercel.app>
- **Stars：** ~1,020（2026-09-06）
- **入库日期：** 2026-09-06
- **一句话说明：** 浏览器 **3D 解剖探索器**：BodyParts3D 4.0 成人男性参考 **2,234** 可选 mesh、**15** 系统层、**3,432** FMA 概念检索与 exploded view；React + Three.js + shadcn/ui。
- **沉淀到 wiki：** 是 → [`wiki/entities/human-atlas.md`](../../wiki/entities/human-atlas.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源** |
| **应用代码** | MIT License — <https://github.com/ashemag/human-atlas> |
| **解剖数据** | BodyParts3D 4.0，**CC BY 4.0**（见 `public/ATTRIBUTION.md`） |
| **演示站** | Vercel 静态部署；无需 API key |

## README 要点（2026-09-06）

### 功能

- 轨道/缩放/点选 **2,234** 独立 mesh
- **15** 解剖系统层开关；骨架/器官 preset
- Assembled ↔ **exploded**  spaced inventory
- 搜索解剖名与 source id；isolate + 详情面板
- 移动端紧凑控制；可选 **WebMCP** 工具（兼容浏览器）

### 数据与性能

| 项 | 数值 |
|----|------|
| 数据源 | BodyParts3D **4.0** 成人男性参考（TARO MRI + 插图细化） |
| 三角面 | **2,288,268**（简化后仍保留全部 source mesh） |
| 压缩几何下载 | ~**33 MB** |
| 命名概念 | **3,432** FMA concepts（可对应多 mesh） |
| 简化 | meshoptimizer **0.2%** 相对误差/结构 |

### 渲染架构

- 几何 **batch merge**；per-structure GPU texture 控平移/可见/选中
- Component geometry 保 accurate picking
- Exploded layout 仅 pack **可见** 块；避免数千 draw call

### 本地运行

```sh
npm ci && npm run dev   # Node ≥22.13 → http://localhost:3016
npm run check && node scripts/validate-atlas.mjs && npm run build
```

### 重建几何（可选）

官方 BodyParts3D OBJ + 英文 metadata → `scripts/convert-anatomy.py` → `optimize-anatomy.mjs` → `compress-models.mjs`

## 对 wiki 的映射

- 实体：[`wiki/entities/human-atlas.md`](../../wiki/entities/human-atlas.md)
- 演示摘录：[`sources/sites/human-atlas-demo.md`](../sites/human-atlas-demo.md)
