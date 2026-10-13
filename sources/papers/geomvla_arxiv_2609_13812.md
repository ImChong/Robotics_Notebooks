# GeomVLA: Unifying Scene, Motion, and Action in 3D

> 来源归档（ingest）

- **标题：** GeomVLA: Unifying Scene, Motion, and Action in 3D
- **类型：** paper
- **机构：** 卡内基梅隆大学（Carnegie Mellon University）；英伟达（NVIDIA）
- **作者：** Ziyin Xiong、Nikolaos Gkanatsios、Moritz Reuss、Katerina Fragkiadaki
- **原始链接：**
  - <https://arxiv.org/abs/2609.13812>
  - <https://arxiv.org/html/2609.13812v2>
  - 项目页：<https://ziyin-xiong.github.io/geomvla.io/>
- **入库日期：** 2026-10-13
- **一句话说明：** GeomVLA 将视觉语言特征提升到机器人中心三维坐标系，在动作生成前预测任务条件的未来场景运动，再以几何感知注意力引导流匹配动作头；重点是让感知、运动中间表征和动作共享几何一致性，而不是把预测轨迹当作开环执行计划。

## 核心论文摘录（MVP）

### 1) 问题与主张：VLA 缺少可度量的三维几何一致性

- **链接：** <https://arxiv.org/html/2609.13812v2#S1>
- **摘录要点：** 互联网视觉语言预训练提供语义，却不直接给出机器人操作所需的度量三维空间。GeomVLA 主张在一个机器人中心 3D 坐标系中连接场景理解、未来运动和动作生成；其消融表明，在表征和动作之间加入几何一致的运动监督，比单独预测未来运动更重要。
- **对 wiki 的映射：** [GeomVLA](../../wiki/entities/paper-geomvla-2609-13812.md) — 记录统一坐标系及“几何一致性”主张，并与 2D 表征路线对照。

### 2) 表征与动作：从多视角 RGB-D 到共享 3D 场景空间

- **链接：** <https://arxiv.org/html/2609.13812v2#S2>
- **摘录要点：** 输入包括多视角 RGB-D、语言指令和机器人本体状态。方法用深度及相机标定把预训练 VLM 特征提升为机器人基座坐标系中的 3D 特征，尽量保留语义；动作以机器人中心的笛卡尔量表示，并可扩展为关节角版本。论文采用冻结 Florence-2 作为视觉语言编码器。
- **对 wiki 的映射：** [GeomVLA](../../wiki/entities/paper-geomvla-2609-13812.md) — 说明相机标定、坐标系与动作参数化的复现边界。

### 3) 运动中间表征：场景轨迹去噪器指导动作流

- **链接：** <https://arxiv.org/html/2609.13812v2#S2>
- **摘录要点：** 任务条件的 3D Scene Trajectory Denoiser 预测未来场景点的运动，得到中间运动 token；几何感知注意力将这些 token 注入 3D flow-based action denoiser。预测轨迹用于条件化动作生成，并非直接执行的开环轨迹计划。
- **对 wiki 的映射：** [GeomVLA](../../wiki/entities/paper-geomvla-2609-13812.md) — 以一张流程图呈现场景运动如何成为动作头的中间监督/条件。

### 4) 结果与消融：3D 一致性比“加一个运动头”更关键

- **链接：** <https://arxiv.org/html/2609.13812v2#S3>
- **摘录要点：** CALVIN ABC-D 平均长度为 **4.624**，优于文中对比的 FLOWER **4.480**、X-VLA **4.430**，且未额外进行机器人动作预训练。RoboTwin2 的 50 项任务总体成功率为 Easy **78.6%**、Hard **76.2%**。消融中完整模型为 **4.624**，去掉运动模块为 **4.508**，2D 版本为 **4.437**，在图像空间预测运动为 **4.085**。这些结果支持三维几何一致性设计，但不同基准的指标不可直接横比。
- **对 wiki 的映射：** [GeomVLA](../../wiki/entities/paper-geomvla-2609-13812.md) — 保存同一评测内的基线与消融差值，避免将跨基准结果混为一谈。

### 5) 真机设置与局限：标定与数据规模仍是关键前提

- **链接：** <https://arxiv.org/html/2609.13812v2#S4>
- **摘录要点：** 真机评测使用 ALOHA 2，双 Azure Kinect DK（前、后视角）与腕部 Intel RealSense D405；8 项桌面任务每项评测 20 次。关节角版本在论文匹配的真机设置中报告 **59.4%**，对照 2D-JA 为 **28.8%**。作者指出方法依赖准确相机标定，训练数据仍属基准规模，没有大规模机器人预训练或跨本体数据；潜在几何运动表征也不等于显式语言推理。
- **对 wiki 的映射：** [GeomVLA](../../wiki/entities/paper-geomvla-2609-13812.md) — 把硬件/相机配置、评估规模和适用边界作为复现检查表。

## 项目页与开源核查

- 项目页描述与论文一致，并链接到模型代码地址 <https://github.com/ziyin-xiong/GeomVLA>。
- **截至 2026-10-13，所链接 GitHub 仓库返回 404**，因此无法确认模型代码、权重或数据集可公开获取；不将作者账号中的项目网站源码仓库视为模型实现。
- 论文与方法以 arXiv HTML v2 为准；项目页用于核对作者提供的公开入口及开源状态。
