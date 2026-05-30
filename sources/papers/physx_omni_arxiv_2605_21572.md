# PhysX-Omni: Unified Simulation-Ready Physical 3D Generation for Rigid, Deformable, and Articulated Objects（arXiv:2605.21572）

> 来源归档（ingest）

- **标题：** PhysX-Omni: Unified Simulation-Ready Physical 3D Generation for Rigid, Deformable, and Articulated Objects
- **缩写：** **PhysX-Omni**
- **类型：** paper / 3D 资产生成 / 物理仿真 / 具身 AI 数据引擎
- **arXiv：** <https://arxiv.org/abs/2605.21572>（PDF：<https://arxiv.org/pdf/2605.21572>）
- **项目页：** <https://physx-omni.github.io/>
- **代码：** <https://github.com/physx-omni/PhysX-Omni>
- **数据集：** <https://huggingface.co/datasets/PhysX-Omni/PhysXVerse>
- **预训练权重：** <https://huggingface.co/PhysX-Omni/PhysX-Omni>
- **作者：** Ziang Cao, Yinghao Liu, Haitian Li, Runmao Yao, Fangzhou Hong, Zhaoxi Chen, Liang Pan, Ziwei Liu（S-Lab NTU、ACE Robotics；通讯 Ziwei Liu）
- **入库日期：** 2026-05-30
- **一句话说明：** 统一 **刚体 / 可变形 / 关节体** 的 **仿真就绪（sim-ready）物理 3D 生成**：基于 **Qwen2.5-VL-7B** 的粗到细 VLM 推理 + **模板化 2D RLE** 高分辨率几何表征 + **TRELLIS** 体素解码；配套 **PhysXVerse**（8.7K+ 资产、2.9K+ 类）与 **PhysX-Bench**（六维物理属性评测）。

## 摘要级要点

- **问题：** 多数 3D 生成偏重外观与几何，缺少 **尺度、材料、affordance、运动学、功能描述** 等物理字段；既有物理 3D 工作多 **只覆盖单一资产类型**（仅关节或仅可变形），且受 **标注数据稀缺** 与 **野外评测缺失** 制约。
- **PhysX-Omni 主张：** 单一框架内统一生成 **刚性、可变形、关节化** 三类 sim-ready 资产；几何用 **面向 VLM 的模板化 RLE** 直接编码高分辨率体素切片，**无需额外 special token / 分割模块**，可与现有 **体素解码器（TRELLIS）** 衔接出高质量网格。
- **生成范式（Fig.2）：** 输入完整或部分遮挡单图 → **全局理解**（类别、语义、绝对尺度、部件层级、物理先验）→ **多轮部件级** 几何与物理属性预测 → 导出 **URDF/XML** 等仿真格式（代码管线）。
- **PhysXVerse：** 基于 **PartVerse** 人体校验分割 + [PhysXGen](https://arxiv.org/abs/2605.05163) 式人机协同标注流水线；**8.7K+** 资产、**2.9K+** 室内外类别（家具、无人机、机器人、车辆、大型场景构件等）；部件数 **1–65**，覆盖简单刚体到复杂关节系统。
- **PhysX-Bench：** 六维——**geometry**（CLIP、3D consistency、visual quality）、**absolute scale**、**material**（自由落体/落水仿真视频 + VLM 判读）、**affordance**、**kinematics**（运动视频 + 先验部件一致性等）、**description**（部件 mask 语义对齐）；评测 VLM 为 **Qwen3.5-122B-A10B**（论文 §3.3）。
- **训练：** 合并 **PhysXNet**、**PhysX-Mobility**、**PhysXVerse**，合计 **42K+** sim-ready 资产；每物体 **25 视角** 条件图；**64×A100**、约 **14 天**、峰值 lr **2e-5**、有效 batch **128**、最大序列 **16384** token。
- **相对基线（表 1/2 摘录）：** 在 PhysXVerse 上 **Kinematic** 约 **0.92** vs PhysX-Anything **0.42**；几何 **CD** 显著低于 PhysXGen / PhysX-Anything；PhysX-Bench **kinematic** 约 **80.72** vs 次优 **65.99**（PhysX-Anything）。
- **下游：** 论文展示 **仿真就绪场景生成** 与 **接触丰富机器人策略学习**；仓库提供 `convert_objects2scene.py`、`applications_scene` 场景管线。

## 对 wiki 的映射

- 沉淀实体页：[`wiki/entities/physx-omni.md`](../../wiki/entities/physx-omni.md)
- 互链参考：[PhysForge](../../wiki/entities/paper-physforge-physics-grounded-3d-assets.md)（同赛道 VLM+扩散物理 3D）、[Articraft](../../wiki/entities/articraft.md)（程序化关节资产）、[SAPIEN](../../wiki/entities/sapien.md)、[Manipulation](../../wiki/tasks/manipulation.md)、[Sim2Real](../../wiki/concepts/sim2real.md)
