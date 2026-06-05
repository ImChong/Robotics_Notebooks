# HomeWorld: A Unified Floorplan-to-Furnished Framework for Generating Controllable, Densely Interactive Whole-Home Scenes（arXiv:2606.06390）

> 来源归档（ingest）

- **标题：** HomeWorld: A Unified Floorplan-to-Furnished Framework for Generating Controllable, Densely Interactive Whole-Home Scenes
- **品牌名：** **Kairos · HomeWorld**（项目页标题；与 kairos-agi 的 Kairos 3.0 世界模型为不同项目）
- **类型：** paper / technical report
- **arXiv：** <https://arxiv.org/abs/2606.06390>（PDF：<https://arxiv.org/pdf/2606.06390.pdf>；项目页 mini PDF：<https://kairos-homeworld.github.io/assets/pdf/homeworld-mini.pdf>）
- **项目页：** <https://kairos-homeworld.github.io/>
- **代码：** <https://github.com/Kairos-HomeWorld/HomeWorld>（截至 ingest：**Coming Soon**）
- **数据集：** 项目页标注 **Coming Soon**（计划发布 300K 矢量平面图 + 5K 全屋 furnished 3D 场景）
- **机构：** Ace Robotics、CUHK MMLab、Shenzhen Loop Area Institute（*Equal contribution；†Project lead）
- **作者：** Wenbo Li, Xiaoliang Ju, Zipeng Qin, Rongyao Fang, Hongsheng Li
- **入库日期：** 2026-06-05
- **一句话说明：** 面向 **具身 AI 仿真** 的 **全屋室内场景生成** 统一框架：从文本 prompt 经 **四阶段可控流水线**（平面图 LLM → 图像驱动分层软装 → VLM 递归修正 → 可操纵小物放置）产出 **sim-ready、全局连贯、平均 >15 个可操作物体/场景** 的多房间 3D 家居；并策展 **30 万级真实住宅矢量平面图** 与 **5K 全屋 furnished 3D** 数据集（强调 **Chinese Style** 户型）。

## 摘要级要点

- **问题：** 现有室内生成多聚焦 **单房间** 或 **孤立子任务**（仅平面图 / 仅软装），全屋场景常缺 **全局连贯性、物理合理性、仿真就绪与可操作物体密度**；大规模高质量 3D 住宅数据仍稀缺。
- **策略：** **组合式分层生成**——先用 **可扩展 2D 平面图先验** 锚定全局结构，再 **渐进实例化** 为带家具与可操纵物体的 3D 环境，而非端到端 scarce-3D 直接合成。
- **数据贡献：** **314K** 经质检的 **全矢量、带丰富 caption** 真实住宅平面图（从约 108 万 raw 图像管线提取）；将发布 **5K sim-ready 全屋 3D**（复杂几何、可操纵物体、完整 3D）。
- **仿真就绪：** 附着 **基础物理属性**、**表面纹理与简单光照**；小物体 **surface-centric** 放置于桌面/台面/柜面等支撑面；项目页展示 **生成场景中的具身交互 demo**。
- **对比主张（论文 Table 1）：** 相对 RPLAN / 3D-FRONT / ProcTHOR / Holodeck / LayoutVLM 等，强调 **whole-home scope + sim-ready + manipulable objects（>15/scene）** 的组合。

## 核心论文摘录（MVP）

### 1) 四阶段全屋生成总览

- **链接：** <https://kairos-homeworld.github.io/> Highlights / Method；<https://arxiv.org/abs/2606.06390> Abstract
- **摘录要点：**
  1. **Stage 1 — Floorplan Generation：** 策展 300K 真实住宅平面图；**K-D tree JSON 表征** + **rich captions** 微调 LLM，实现 prompt 条件化 **全屋** 布局生成。
  2. **Stage 2 — Image-Driven Hierarchical Furnishing：** 由平面图实例化 **unfurnished 3D shell（Blender）** 作显式 3D 约束；**top-down roaming** 初始化大件家具 → **ego-centric roaming** 补全细部；**图像 inpainting + SAM-3 / SAM-3D**  grounding 与资产检索。
  3. **Stage 3 — Recursive Refinement：** **微调 VLM refiner** 在 top-view + 结构化 3D layout 上 **迭代检测碰撞、挡门、越界** 并输出 **平移/旋转等 corrective action**，直至通过验证或达迭代上限。
  4. **Stage 4 — Manipulable Object Placement：** 在指定支撑面上 **surface-centric** 放置小物体，服务 embodied AI 仿真。
- **对 wiki 的映射：**
  - [HomeWorld（全屋 sim-ready 场景生成）](../../wiki/entities/paper-homeworld-whole-home-scene-generation.md) — 流程总览与 Mermaid。
  - [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) — 与 **像素级视频 WM** 互补的 **静态 3D 仿真资产** 路线。

### 2) 平面图数据集与 K-D tree LLM

- **链接：** arXiv §3.1；项目页 dataset teaser
- **摘录要点：** 五步法：采集 → 检测墙/门/窗/尺寸链 → OCR 房间标签 → 质检 → 拓扑/连通性标注与 **自动 caption**；训练时将 caption 组件随机组合成多样 prompt。**K-D tree** 输出空间比直接预测多边形更 **结构化、可约束、避免 overlap**。
- **对 wiki 的映射：**
  - [HomeWorld](../../wiki/entities/paper-homeworld-whole-home-scene-generation.md) — 数据引擎与平面图生成节。
  - [视觉–语言导航（VLN）](../../wiki/tasks/vision-language-navigation.md) — 大规模 **中文住宅户型** 平面图对 **室内导航/任务仿真** 数据本地化的意义。

### 3) 分层 roaming + 3D shell 锚定

- **摘录要点：** 空壳几何约束 **非矩形房间** 与 **跨视角 3D 一致性**，缓解 Text2Room 类 **free roaming 几何漂移**；top-down 标注门/窗/邻室关系后再 inpainting；ego 阶段用 **heatmap 贪心视点选择** 覆盖 floor grid。
- **对 wiki 的映射：**
  - [HomeWorld](../../wiki/entities/paper-homeworld-whole-home-scene-generation.md) — Stage 2 机制表。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — **2D foundation prior + 显式 3D 约束** 的组合范式。

### 4) VLM 递归修正与 manipulable 物体

- **摘录要点：** Refiner 训练数据含 **受控 corruption**（碰撞、挡门、越界、浮空等）+ **oracle repair action** + **model-in-the-loop** 样本；Stage 4 面向 **>15 manipulable objects/scene** 的 embodied 交互密度。
- **对 wiki 的映射：**
  - [HomeWorld](../../wiki/entities/paper-homeworld-whole-home-scene-generation.md) — 修正环与仿真就绪节。
  - [Manipulation](../../wiki/tasks/manipulation.md) — sim-ready **可操作物体** 与操作仿真数据链。

### 5) 数据集规模与行业语境（项目页 + 公开报道）

- **摘录要点：** **300K Chinese Style** 矢量平面图；**5K** 全屋 3D；Ace Robotics（大晓机器人）等报道强调 **封闭式厨房、独立生活阳台** 等 **中国家庭户型** 特征，以及 **跨房间导航、多房间整理** 等家务仿真应用（定性，非论文主定量指标）。
- **对 wiki 的映射：**
  - [HomeWorld](../../wiki/entities/paper-homeworld-whole-home-scene-generation.md) — 数据集与局限。
  - [Sim2Real](../../wiki/concepts/sim2real.md) — **仿真场景规模化** 相对实地采集的成本优势（叙事层）。

## BibTeX（项目页提供）

```bibtex
@misc{li2026homeworld,
  title         = {HomeWorld: A Unified Floorplan-to-Furnished Framework for Generating Controllable, Densely Interactive Whole-Home Scenes},
  author        = {Wenbo Li and Xiaoliang Ju and Zipeng Qin and Rongyao Fang and Hongsheng Li},
  year          = {2026},
  eprint        = {2606.06390},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2606.06390},
  note          = {Project page: https://kairos-homeworld.github.io/}
}
```

## 对 wiki 的映射

- 沉淀实体页：[`wiki/entities/paper-homeworld-whole-home-scene-generation.md`](../../wiki/entities/paper-homeworld-whole-home-scene-generation.md)
- 项目页归档：[`sources/sites/kairos-homeworld-github-io.md`](../sites/kairos-homeworld-github-io.md)
- 代码索引：[`sources/repos/homeworld.md`](../repos/homeworld.md)

## 相邻工作（论文 Related Work，未单独 ingest）

- Holodeck、LayoutVLM、ProcTHOR、Infinigen、PhyScene、EmbodiedGen、Text2Room、Floorplan-LLaMA 等 — 见 arXiv Table 1 对比
- PhysX-Omni / PhysForge（本库已有）— **物体/资产级** sim-ready 生成；HomeWorld 侧重 **全屋布局 + 场景实例化**

## 当前提炼状态

- [x] 摘要、四阶段流水线、K-D tree 平面图、roaming/refiner/manipulable 叙事
- [x] 项目页 / arXiv / GitHub（占位）入口
- [ ] 代码与数据集公开发布后补 HF/下载链接与定量表
- [ ] 细读 PDF 后补充 user study 指标与基线命名
