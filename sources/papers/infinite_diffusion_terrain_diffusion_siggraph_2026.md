# InfiniteDiffusion: Bridging Learned Fidelity and Procedural Utility for Open-World Terrain Generation

> 来源归档（ingest）

- **标题：** InfiniteDiffusion: Bridging Learned Fidelity and Procedural Utility for Open-World Terrain Generation
- **类型：** paper / procedural-terrain / diffusion / world-generation / siggraph
- **来源：** SIGGRAPH 2026 Conference Papers（ACM）
- **原始链接：**
  - 项目页：<https://xandergos.github.io/terrain-diffusion/>
  - arXiv：<https://arxiv.org/abs/2512.08309>
  - DOI：<https://doi.org/10.1145/3799902.3811080>
  - 代码：<https://github.com/xandergos/terrain-diffusion>
- **作者：** Alexander Goslin（Independent Researcher）
- **入库日期：** 2026-07-14
- **一句话说明：** 提出 **InfiniteDiffusion**——无需再训练的算法，把任意扩散采样改造成 **无限、惰性、O(1) 随机访问、种子确定** 的无界张量；并以 **Terrain Diffusion** 演示 **学习式程序化地形**：分层扩散堆栈 + Laplacian 编码覆盖地球尺度高程动态范围，在消费级 GPU 上以约 **9× 轨道速度** 实时流式生成，并开源 **Minecraft Fabric** 与 Unity 集成。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 代码仓库 | <https://github.com/xandergos/terrain-diffusion> | Python 推理、探索 GUI、API、TIFF 导出 |
| Infinite Tensor | <https://github.com/xandergos/infinite-tensor> | 常数内存操作无界张量的底层框架 |
| Minecraft Mod | <https://github.com/xandergos/terrain-diffusion-mc> · [Modrinth](https://modrinth.com/mod/terrain-diffusion) | Fabric 世界类型替换；`/td-explore` 内置地形浏览器 |
| Hugging Face 模型 | <https://huggingface.co/collections/xandergos/terrain-diffusion> | `terrain-diffusion-30m`（可玩尺度）/ `terrain-diffusion-90m`（大尺度写实） |
| 入门 demo | `annotated_infinite_panorama.py` | 仅用 Stable Diffusion v1.5 + infinite-tensor 复现 InfiniteDiffusion 核心 |

## 摘要级要点

### 1) 三难困境与 InfiniteDiffusion

- **经典困境：** 内容生成长期面临 **无限延展 / 无状态生成 / 学习式逼真** 三选二——扩散模型逼真但 **有界**；经典 Perlin 等噪声 **无限且确定性** 但 **不可学习**；自回归外绘可学习无界内容，却依赖 **全局共享状态**，破坏 **确定性** 与 **随机访问**。
- **InfiniteDiffusion：** **training-free** 地把 MultiDiffusion 推广到 **无限或超内存域**，把扩散过程重表述为 **惰性计算**——只在你请求的坐标区域采样；仅依赖 **seed + 坐标** 索引，**O(1) 随机访问**、**全序不变确定性**、**可 embarrassingly parallel**；内部 LRU 缓存仅为性能优化，**无持久外部状态**。
- **相对自回归外绘：** 随机访问 O(1) vs O(n)；确定性 **顺序无关** vs 顺序依赖；**无误差累积** vs 复合漂移；**可并行** vs 串行；**training-free** vs 需专门训练。

### 2) Terrain Diffusion 应用栈

- **定位：** 首个 **学习式程序化地形生成器**，接口类似程序化噪声（seed + 坐标查询高程/气候），但具备扩散模型保真度。
- **尺度：** Laplacian 编码稳定 **-10000 m（马里亚纳海沟）～ +9000 m（珠峰）** 的垂直动态范围；**级联扩散** 耦合行星尺度上下文与局部细节，单张 1024×1024 relief 可覆盖 **100 km** 宽度，大陆可达 **百万 km²**。
- **两阶段管线：** **粗图（coarse map）** 可由程序化草图或手绘（如 Azgaar JSON）提供布局；**细模型** 将粗条件 refine 为 **30 m/px 或 90 m/px** 高程/气候场，并可 **256×** 分辨率上采样导出 GeoTIFF。
- **性能：** 消费级 GPU 上地形生成速度约为 **轨道速度的 9 倍**；Unity 演示中玩家以约 **3× 轨道速度** 舒适飞行；Minecraft 多人可 **种子共享、瞬移百万格**。

### 3) 模型与工程入口（README 归纳）

| 模型 | 分辨率 | 适用 |
|------|--------|------|
| `xandergos/terrain-diffusion-30m` | 30 m/px，粗像素 ~7.7 km | **可玩世界**；局部变化更丰富，推荐游戏/交互 |
| `xandergos/terrain-diffusion-90m` | 90 m/px，粗像素 ~23 km | **大尺度写实**；连贯性更强，适合宏观世界构建 |

- **CLI：** `python -m terrain_diffusion explore <model>` 双面板 GUI；`api` 子命令提供高程/气候查询 API；`azgaar-to-tiff` / `tiff-export` 支持 Azgaar 条件图 → GeoTIFF。
- **高级定制：** `synthetic_map.py` 修改程序化粗图；可重训 tiny coarse 模型（ETOPO + WorldClim）。

### 4) Minecraft Mod（Modrinth / terrain-diffusion-mc）

- **集成：** Fabric 世界类型 **Terrain Diffusion**；首次在线下载模型 **~2.5 GB**；VRAM **~1.5 GB**（`inference.offload_models=true` 时峰值），系统 RAM 建议 **≥2.5 GB** 分配给 Minecraft。
- **World Scale（1–6）：** 控制每 block 代表的真实米数（scale=1 → 30 m/block；scale=2 → 15 m/block，推荐）；影响世界高度与 GPU/CPU 瓶颈权衡。
- **Explorer：** `/td-explore` 启动本地 Web UI（默认端口 19801），可浏览大陆/山脉/岛屿并取坐标。

## 核心摘录（面向 wiki 编译）

### 1) InfiniteDiffusion vs MultiDiffusion vs Auto-Regression

- **链接：** <https://xandergos.github.io/terrain-diffusion/>
- **摘录要点：** MultiDiffusion 在 **预定义有界画布** 上 eager 生成；InfiniteDiffusion **不施加边界**，质量相对 MultiDiffusion **几乎无退化**，同时获得无限、无状态、惰性三大性质。相对自回归：O(1) 随机访问、顺序不变确定性、无误差复合、可并行、training-free。
- **对 wiki 的映射：**
  - [InfiniteDiffusion / Terrain Diffusion](../../wiki/entities/paper-infinite-diffusion-terrain-diffusion.md) — 算法与对比表
  - [Procedural Terrain Generation](../../wiki/concepts/procedural-terrain-generation.md) — 学习式后继路线

### 2) 分层扩散 + Laplacian 编码 + infinite-tensor

- **链接：** <https://xandergos.github.io/terrain-diffusion/> · <https://github.com/xandergos/terrain-diffusion>
- **摘录要点：** 行星上下文与局部细节由 **分层扩散堆栈** 耦合；**紧凑 Laplacian 编码** 稳定地球尺度高程动态范围；**infinite-tensor** 框架实现无界张量的常数内存操作与惰性采样。
- **对 wiki 的映射：**
  - [InfiniteDiffusion / Terrain Diffusion](../../wiki/entities/paper-infinite-diffusion-terrain-diffusion.md) — Mermaid 管线与工程实践
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 扩散生成式环境资产（非像素视频 rollout）

### 3) 游戏引擎落地：Minecraft + Unity

- **链接：** <https://modrinth.com/mod/terrain-diffusion> · <https://github.com/xandergos/terrain-diffusion-mc>
- **摘录要点：** 作为 **开源 Minecraft Fabric mod** 发布，无外部依赖；种子共享、多人、瞬移；证明 **函数式无状态** 接口可嵌入任意引擎。Unity 技术演示以 **3× 轨道速度** 飞行。
- **对 wiki 的映射：**
  - [InfiniteDiffusion / Terrain Diffusion](../../wiki/entities/paper-infinite-diffusion-terrain-diffusion.md) — 部署与资源需求
  - [Legged Gym](../../wiki/entities/legged-gym.md) / [Procedural Terrain Generation](../../wiki/concepts/procedural-terrain-generation.md) — 机器人仿真侧程序化地形对照

## 对 wiki 的映射

- 主实体页：**`wiki/entities/paper-infinite-diffusion-terrain-diffusion.md`**
- 概念交叉：**`wiki/concepts/procedural-terrain-generation.md`**（噪声/高度场 → 学习式扩散程序化）
- 方法交叉：**`wiki/methods/generative-world-models.md`**（生成式环境层：高程场而非视频 WM）

## 引用（项目页 BibTeX）

```bibtex
@inproceedings{goslin2026infinitediffusion,
  author    = {Goslin, Alexander},
  title     = {InfiniteDiffusion: Bridging Learned Fidelity and Procedural Utility for Open-World Terrain Generation},
  booktitle = {Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  year      = {2026},
  pages     = {10 pages},
  publisher = {ACM},
  address   = {New York, NY, USA},
  doi       = {10.1145/3799902.3811080},
  url       = {https://doi.org/10.1145/3799902.3811080},
  series    = {SIGGRAPH Conference Papers '26}
}
```
