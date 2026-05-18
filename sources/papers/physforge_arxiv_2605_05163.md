# PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World（arXiv:2605.05163）

> 来源归档（ingest）

- **标题：** PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World
- **缩写：** **PhysForge**
- **类型：** paper / 3D 资产生成 / 物理交互 / 具身 AI 仿真数据
- **arXiv：** <https://arxiv.org/abs/2605.05163>（PDF：<https://arxiv.org/pdf/2605.05163>）
- **项目页：** <https://hku-mmlab.github.io/PhysForge/>
- **作者：** Yunhan Yang, Chunshi Wang, Junliang Ye, Yang Li, Zanxin Chen, Zehuan Huang, Yao Mu, Zhuo Chen, Chunchao Guo, Xihui Liu（HKU、腾讯混元、ZJU、THU、SJTU、BUAA 等；\* 同等贡献，通讯作者以论文为准）
- **入库日期：** 2026-05-18
- **一句话说明：** 面向 **可交互虚拟世界与具身智能** 的 **物理接地 3D 资产生成**：先由 **VLM** 输出 **分层物理蓝图**（材料、功能、运动学约束），再由 **物理接地扩散模型** 在 **KineVoxel Injection（KVI）** 机制下 **联合** 合成高保真几何与 **精确关节参数**；配套 **PhysDB**（约 **15 万** 资产、**四档** 物理标注）支撑训练与评测。

## 摘要级要点（与 abs / 方法总览一致）

- **问题：** 多数 3D 生成只做 **静态几何与外观**，缺少 **功能与层级物理**，产出的「空壳」资产难以在需要 **抓取、推动、关节操作** 的仿真或游戏中直接使用。
- **主张：** 可物理交互对象的生成应由 **功能逻辑 + 分层物理** 驱动，而非单纯整体外形拟合。
- **PhysForge 两阶段：**
  1. **VLM 规划（Physical Architect）：** 输入单视图图像、可选 2D mask、以及粗 3D 体素（论文语境中与 **TRELLIS** 一类管线衔接），自回归生成 **Hierarchical Physical Blueprint**：部件包围盒布局、父子关系、材料/质量、内在功能与状态机、交互层（原子 affordance、关节类型与参数语义）等。
  2. **扩散实现：** 在 **OmniPart** 二阶段框架上扩展 **KVI**：将每部件关节参数 \(O_i, A_i, L_i\)（原点、轴、限位）编码为可与几何潜变量 **同空间去噪** 的 **KineVoxel**，经独立轻量编解码器注入主干 Transformer，并用 **VLM 给出的关节类型嵌入** 区分几何 token 与运动学 token；训练目标为 **条件流匹配（CFM）** 下的几何与运动学分项 \(L_2\) 速度损失（论文报告 \(\lambda_{\text{kine}}=10\) 以强调关节精度）。
- **PhysDB：** 从 **Objaverse** 等来源筛选可部件化对象，经 **多模态 LLM 初标 + 人工筛校** 得到约 **150k** 样本；标注分 **整体 / 静态 / 功能 / 交互** 四层级；大规模场景下 **精确数值关节轴** 标注困难，训练阶段辅以 **PartNet-Mobility**、**Infinite-Mobility** 等提供 **真值关节参数** 以补运动学监督。
- **实验叙事：** 在 **PhysXNet**、自建 **PhysDB** 测试子集、**PartObjaverse-Tiny** 规划任务及与 **Articulate Anything** 等基线的 **关节轴/枢轴误差** 对比上报告优势；定性展示导入 **RoboTwin** 类操作仿真与 **Unity/UE** 等虚拟世界的 **simulation-ready** 资产。

## 对 wiki 的映射

- 沉淀实体页：[`wiki/entities/paper-physforge-physics-grounded-3d-assets.md`](../../wiki/entities/paper-physforge-physics-grounded-3d-assets.md)
- 互链参考：[Articraft](../../wiki/entities/articraft.md)（程序化可关节资产生成对照）、[RoboTwin](../../wiki/entities/robotwin.md)（论文下游演示语境）、[Sim2Real](../../wiki/concepts/sim2real.md)、[SAPIEN](../../wiki/entities/sapien.md)
