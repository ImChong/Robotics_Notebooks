# Harvard Computational Robotics — SceneAgent

> 来源归档（ingest）

- **标题：** SceneAgent: 3D Capture-Derived Scenes with Predictive Physics for Policy Evaluation and Training Environments
- **类型：** site（官方项目页）
- **发布方：** Harvard Computational Robotics Group，School of Engineering and Applied Sciences，Harvard University
- **原始链接：** <https://computationalrobotics.seas.harvard.edu/SceneAgent/>
- **配套论文：** Preprint（2026）；归档见 [sources/papers/sceneagent_harvard_preprint_2026.md](../papers/sceneagent_harvard_preprint_2026.md)
- **代码：** <https://github.com/ComputationalRobotics/SceneAgent> — 归档见 [sources/repos/computationalrobotics-sceneagent.md](../repos/computationalrobotics-sceneagent.md)（**截至 2026-09-27 仅为项目页静态导出**）
- **入库日期：** 2026-09-27
- **最近复核：** 2026-09-27
- **一句话说明：** **Agentic Real2Sim** 管线：3DGS / 摄影测量 / LiDAR 捕获 → 语义 + **per-Gaussian 预测物理** + 分解/关节化 + **Digital Sisters** → **USDZ** 导出至 Isaac Lab、MuJoCo、Unreal；含 **demonstration factory + VLA LoRA 微调** 闭环；完整评测数字仍在更新中。

## 摘录要点（与论文分工）

- **TL;DR：** 把真实 3D 捕获转成带预测物理、可交互的仿真环境，用于策略评测、在线规划与 **纯仿真训练/微调**。
- **输入模态：** RGB 序列（可经 COLMAP + NerfStudio `splatfacto` 训 3DGS）、既有 `.ply` 3DGS、摄影测量、LiDAR；亦支持 World Labs Marble 等 **生成式 3DGS** 场景。
- **八步管线（页面）：** 3DGS → GroundingDINO+SAM 语义特征（LangSplat 风格 codebook）→ per-Gaussian 预测物理 + VLM 摩擦/刚性/质量 → 物体分解 → 关节化 → **object sisters** → USDZ 打包 → Three.js 交互编辑 viewer。
- **Digital Sisters：** 相对 digital twin 的 **非精确复刻变体**（几何/视觉微扰 + 布局/光照随机化），用于 sim→real 泛化；页面称 initial testing 提升真机完成率，完整 ablation 进行中。
- **混合场景：** 单物体 SceneAgent 资产 + Omniverse / Isaac Lab 艺术家背景（RoboCasa 类厨房、机房、化学台等）。
- **Agentic 编排：** 多 agent 处理异构格式与易错分割/尺度/朝向；可在 Isaac Lab 内渲染 **视觉审查** 并自动修正。
- **策略训练块：** 不可交互 Gaussian 场景 → **demonstration factory**（初始状态随机 + 物理沉降 + 脚本专家 pick-place + success gate）→ 深度合成观测（mesh 前景 + splat 背景）→ **LoRA 微调** π₀.₅ / GR00T 1.6 等 VLA；页面称 sim-only 微调后真机有提升，**完整评测待更新**。
- **Embodiment：** 主推 Franka + 改型 DROID/UMI 夹爪；展示 Sharpa 灵巧手、Dexmate 移动双臂等。

## 论文 / 代码状态

- 项目页 Citation：`@misc{sceneagent2026, … note={Preprint. …}}`；**未列 arXiv URL**（截至入库日）。
- 页脚：**「The code, processed scenes, and example converted objects will be released with our paper soon.」**
- GitHub [ComputationalRobotics/SceneAgent](https://github.com/ComputationalRobotics/SceneAgent) 描述为 **「SceneAgent project website (static export)」** → **宣称将开源 / 待发布**（管线与示例资产未随仓）。

## 对 wiki 的映射

- [SceneAgent 论文实体](../../wiki/entities/paper-sceneagent-real2sim-capture-physics.md) — 管线、digital sisters、训练闭环与开源边界
- [官方仓归档](../repos/computationalrobotics-sceneagent.md) — 静态站仓 vs 未来管线仓
- [Sim2Real](../../wiki/concepts/sim2real.md) — Real2Sim 资产与 sim2real 训练
- [SimFoundry 论文实体](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md) — 页面自比评测与 cousins/sisters 对照
- [Agentic Real2Sim](../../wiki/entities/paper-agentic-real2sim.md) — 另一条 VLM-agent Real2Sim 编排路线
