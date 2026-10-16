# GATOR: Generative and Agentic 3D Object Reconstruction From Casual Images（arXiv:2610.11215）

> 来源归档（ingest）— 根据 arXiv 摘要及作者官方项目页整理，不是原文镜像。

- **论文标题：** GATOR: Generative and Agentic 3D Object Reconstruction From Casual Images
- **类型：** 3D 对象重建 / 场景对齐 / 仿真就绪资产
- **作者：** Qirui Wu、Stan Birchfield、Hesam Rabeti、Angel X. Chang、Bowen Wen
- **机构：** NVIDIA；西蒙菲莎大学（Simon Fraser University）
- **论文：** [arXiv 摘要](https://arxiv.org/abs/2610.11215) · [PDF](https://arxiv.org/pdf/2610.11215)
- **项目页：** <https://research.nvidia.com/labs/lpr/gator/>
- **代码：** **待发布**（项目页标注 `Code (Coming Soon)`，截至 2026-10-16 未提供代码仓链接）
- **提交日期：** 2026-10-08
- **一句话说明：** 从一张或多张随手拍图像恢复有纹理、保留场景相对位姿的完整 3D 物体；先用多模态生成建立资产，再由图像证据引导的 agent 局部修复结构与纹理。

## 论文要点

1. **稀疏观察下的场景对齐重建：** RGB、目标 mask 与 pointmap 特征按 patch 对齐，再做跨视角推理；目标是在遮挡和杂乱场景中补全物体，同时保留场景相对位置、朝向和尺度。
2. **文本条件化生成：** 类别文本辅助稀疏结构生成；对象描述与投影图像特征进一步引导几何、外观和纹理生成，并使用分阶段 adapters。
3. **证据驱动的 agent 精修：** 生成资产作为初始化，agent 在 Blender 中执行局部编辑、渲染与对照审阅，修复细节结构、拓扑、缺失部分和纹理；保留有证据支持的修改，撤销失败修改。
4. **仿真侧验证：** 项目页展示室内场景重建、Isaac Sim 中 46 个物体组成的房间受重力沉降，以及人形机器人与重建场景交互的演示。
5. **时间预算结果：** 在 10 个抽样 Toys4K 对象上，生成阶段约 30 秒；论文项目页报告其在 Chamfer Distance 和 LPIPS 上优于从零开始、获得 20 分钟预算的 agent。该比较是作者页面所列设定，不应外推为所有场景的普遍时延或质量结论。

## 开源与复现状态

- 截至 **2026-10-16**，项目页的代码状态为 **Coming Soon**，未列出 GitHub、权重或数据集下载地址；目前可核对方法说明、交互重建展示与视频演示。
- 因尚无可运行官方实现，源码运行时序图不适用。后续有代码发布时，再核实生成器、Blender agent、资产导出及 Isaac Sim 场景构建入口。

## 对 wiki 的映射

- [GATOR 详情页](../../wiki/entities/paper-gator.md)
- [PhysForge](../../wiki/entities/paper-physforge-physics-grounded-3d-assets.md)（比较图像重建资产与单视图生成可关节资产）
- [Sim2Real](../../wiki/concepts/sim2real.md)
