# Arnis（真实地理 → Minecraft 世界生成）

> 来源归档

- **标题：** arnis
- **类型：** repo
- **链接：** https://github.com/louis-e/arnis
- **官网：** https://arnismc.com/
- **作者：** Louis Erbkamm（louis-e）
- **许可：** Apache-2.0（`src/luanti_block_map.rs` 含 MC2MT 衍生 LGPL-2.1+ 片段）
- **Stars：** ~17.7k（2026-09-07）
- **技术栈：** Rust、Tauri GUI、OpenStreetMap / Overpass、高程数据集、Overture（可选建筑）
- **入库日期：** 2026-09-07
- **一句话说明：** 从 OpenStreetMap + 真实高程生成 Minecraft Java（1.17+）、Bedrock 与 Luanti（Minetest）世界的开源工具；支持 GUI 框选或 CLI `--bbox`，三种生成模式（geo-terrain / geo-only / terrain-only）。
- **沉淀到 wiki：** [arnis](../../wiki/entities/arnis.md)

---

## 开源状态（2026-09-07 项目页核查）

| 组件 | 状态 |
|------|------|
| GitHub 源码 | **已开源**（Apache-2.0） |
| 预编译发行版 | GitHub Releases（Windows / macOS / Linux） |
| 官方下载渠道 | **仅** [arnismc.com](https://arnismc.com/) 与 GitHub；README 警告第三方镜像可能恶意 |
| 云服务 | [MapSmith](https://arnismc.com/mapsmith/) — 浏览器端生成（非本仓源码） |
| 数据依赖 | OSM（Overpass）、公开高程（AWS 博客提及大规模 elevation 管线） |

---

## 核心摘录

### 1) 数据 → 方块世界管线

- 拉取 **OpenStreetMap** 矢量（道路、建筑等）与 **高程** 数据，映射为 Minecraft 方块与结构。
- GUI：地图矩形选区 + 选择已有世界目录 → **Start Generation**；可配比例、出生点、建筑内部等。
- CLI 示例：
  ```bash
  cargo run --release --no-default-features -- \
    --output-dir="C:/YOUR_PATH/.minecraft/saves/worldname" \
    --bbox="min_lat,min_lng,max_lat,max_lng"
  ```
- Nix：`nix run github:louis-e/arnis -- --output-dir=... --bbox="..."`

**对 wiki 的映射：** [arnis](../../wiki/entities/arnis.md)、[procedural-terrain-generation](../../wiki/concepts/procedural-terrain-generation.md)

### 2) 生成模式（`--mode`）

| 模式 | 结果 |
|------|------|
| `geo-terrain`（默认） | OSM 建筑/道路 + 真实高程地形 |
| `geo-only` | OSM 对象铺在平地 |
| `terrain-only` | 仅高程地形，跳过 OSM/Overture 查询 |

**对 wiki 的映射：** [arnis](../../wiki/entities/arnis.md)

### 3) 目标平台与生态

- **输出：** Minecraft Java 1.17+、Bedrock、Luanti（Minetest）
- **文档：** [GitHub Wiki](https://github.com/louis-e/arnis/wiki/)
- **学术/媒体：** AWS Public Sector 博客、Hackaday、Tom's Hardware 等（README 列表）

**对 wiki 的映射：** [arnis](../../wiki/entities/arnis.md)、[drive-game](../../wiki/entities/drive-game.md)（同为 OSM+高程真实地理管线）

---

## 与机器人研究的邻接读法

- **不是** RL 仿真器或物理引擎——无关节动力学、无传感器 API。
- 价值在 **真实世界 GIS → 可探索 3D 体素环境**：教育/灾害演练（如 Floodcraft 论文引用）、开放世界 agent 的 **地理锚定关卡**、与 [Terrain Diffusion](../papers/infinite_diffusion_terrain.md) 等 **Minecraft 程序化地形** 互补（本工具偏 **测绘数据忠实复刻**，扩散 mod 偏 **学习式无限延展**）。
- 若用于腿式/导航研究，仍需 **碰撞体对齐、摩擦 DR、课程调度**——方块世界 ≠ sim-ready 接触模型。

---

## 对 wiki 的映射

- **wiki/entities/arnis.md** — 独立实体页（本 ingest 新建）
- **wiki/concepts/procedural-terrain-generation.md** — 真实地理数据源交叉引用
- **wiki/entities/drive-game.md** — OSM+DEM 管线对照

## 当前提炼状态

- [x] README、官网与 Releases 核查
- [x] 开源状态（Apache-2.0 + 官方下载警告）
- [x] wiki 实体页
