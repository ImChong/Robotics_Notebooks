# World Labs Blog：Streaming 3DGS worlds on the web（Spark 2.0）

> 官方技术博客来源归档（ingest）

- **标题：** Streaming 3DGS worlds on the web
- **类型：** blog / official
- **URL：** <https://www.worldlabs.ai/blog/spark-2.0>
- **作者/机构：** World Labs
- **日期：** 2026-04-14（文内标注）
- **入库日期：** 2026-05-28
- **一句话说明：** Spark 2.0 技术深稿：**连续 LoD splat 树**、**.RAD 渐进流式**、**GPU 虚拟 splat 分页**；全局 splat 生成→排序→渲染三阶段；Rust→Wasm 在 Web Worker 做 LoD 遍历；与 THREE.js / WebGL2 对齐。

## Spark 1.x 系统回顾（博客归纳）

1. **全局 splat 列表**：多 3DGS 对象变换到同一坐标系，GPU 上可编程管线（GLSL / shader graph）做重着色、裁剪、4DGS 插值等。
2. **排序**：GPU 算距离 → 读回 CPU → 后台 Web Worker **两遍基数排序**；支持多视点并行排序。
3. **渲染**：单次 instanced draw；每 splat 为包围 2D 椭圆的四边形，fragment 评估 Gaussian 不透明度后硬件 blend。

**动机：** 旧 Web 渲染器多 **单对象**、难 **4DGS**、框架小众或强依赖 WebGPU；Spark 选 THREE.js + WebGL2 以覆盖桌面 / iOS / Android / VR。

## Spark 2.0 三大扩展

| 技术 | 作用 |
|------|------|
| **Level-of-Detail（LoD splat 树）** | 连续 LoD：内部节点由子 splat 合并；按视锥与 **splat budget**（约 500K–2.5M）做优先队列遍历，O(budget) 与总树规模解耦 |
| **Progressive Streaming（.RAD）** | 新格式：JSON 元数据 + 64K splat 可随机 seek 的 **RADC** 块；首块 ~64K 最粗 splat 近即时显示；按视点重算 chunk 优先级，3 个 Worker 并行拉取 |
| **Virtual Memory（splat 分页）** | GPU 固定 **16M splat** 池 + 64K 页表；LRU 换入换出 .RAD chunk；多文件共享同一页表 |

### LoD 建树算法

- **Tiny-LoD**（Web 默认）：可变步长体素网格 + 排序数组分桶（借鉴基因组学 k-mer 技巧）；默认基数 **1.6**（非 2 的 octree）以减 popping。
- **Bhatt-LoD**（CLI `build-lod` 默认）：Bhattacharyya 距离合并相似形状/颜色的 splat 对；树规模约原 splat 数的 **130–40%**。

### .RAD vs .PLY / .SPZ

- **.PLY**：行序、可渐进但体积大（~10M splat SH0–3 可达 ~2.3 GB）。
- **.SPZ**：列序压缩好（~200–250 MB / 10M）但须整文件解压才能读全属性。
- **.RAD**：`RAD0` 头 + 分块 RADC；列序 + 每属性可配置编码 + gzip；支持 **缺 chunk 时渲染父节点** 直至子节点到达。

### 其它工程细节

- **Foveated splats**：`coneFov0` / `coneFov` / `coneFoveate` / `behindFoveate` 控制视向与背后的细节预算。
- **Composite LoD**：多棵 LoD 树实例同时入队，统一预算下全局选 splat。
- **案例**：Starspeed（1 亿+ splat，.rad）、Marble 生成世界等。

## 对 wiki 的映射

- [`wiki/entities/spark-3dgs-renderer.md`](../../wiki/entities/spark-3dgs-renderer.md) — Spark 开源渲染器技术页
- [`wiki/entities/world-labs.md`](../../wiki/entities/world-labs.md) — 公司/产品上下文
- [`wiki/comparisons/spark-vs-aholo-web-3dgs-renderers.md`](../../wiki/comparisons/spark-vs-aholo-web-3dgs-renderers.md) — 与 Aholo 对照

## 当前提炼状态

- [x] Spark 2.0 系统架构与格式要点
- [x] 与 wiki 实体/对比页交叉索引
