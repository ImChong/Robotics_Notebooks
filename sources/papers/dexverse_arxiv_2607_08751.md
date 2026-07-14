# DexVerse: A Modular Benchmark for Multi-Task, Multi-Embodiment Dexterous Manipulation

> 来源归档（ingest）

- **标题：** DexVerse: A Modular Benchmark for Multi-Task, Multi-Embodiment Dexterous Manipulation
- **类型：** paper / dexterous-manipulation / benchmark / imitation-learning / multi-embodiment / visuomotor / isaac-lab
- **arXiv abs：** <https://arxiv.org/abs/2607.08751>
- **提交日期：** 2026-07-09
- **项目页：** <https://ycyao216.github.io/DexVerse.site/>
- **机构：** UNC-Chapel Hill、The University of Hong Kong、UC Berkeley
- **作者：** Yunchao Yao\*、Zhuxiu Xu\*（共同一作）、Tianqi Zhang、Zixian Liu、Sikai Li、Zhenyu Wei、Feng Chen、Dihong Huang、Kechang Wan、Chenyang Ma、Shuqi Zhao、Shenghua Gao、Masayoshi Tomizuka、Yi Ma、**Mingyu Ding†**（通讯作者）
- **仿真：** Isaac Lab（manager-based env）；配置驱动任务模板 + 可插拔臂–手具身
- **入库日期：** 2026-07-14
- **一句话说明：** 发布 **100** 项灵巧操作任务、**3** 臂 × **6** 手多具身、可配置视觉域随机与 **3,180** 条 VR 遥操作示范（本体/RGB/深度/点云/状态）；在 **19** 项 baseline 上评测 Diffusion Policy、DP3、OpenVLA、π₀.₅，最佳平均成功率仅 **34%**，暴露跨任务泛化与细粒度接触对齐瓶颈。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://ycyao216.github.io/DexVerse.site/> | 任务可视化、数据与 baseline 说明 |
| 同类灵巧 benchmark | [DexJoCo](https://arxiv.org/abs/2407.00187)、[DexMimicGen](https://arxiv.org/abs/2410.03441)、[Bi-DexHands](https://arxiv.org/abs/2206.11176) | 任务/示范/多具身维度对照（Table 1） |
| 夹爪长程 benchmark | [CALVIN](https://arxiv.org/abs/2112.03227)、[LIBERO](https://arxiv.org/abs/2306.03378)、[RoboTwin 2.0](https://arxiv.org/abs/2506.18088) | 侧重平行夹爪，非高 DoF 手 |
| 策略基线 | [Diffusion Policy](https://arxiv.org/abs/2303.04137)、[DP3](https://arxiv.org/abs/2406.01584)、[OpenVLA](https://arxiv.org/abs/2406.09220)、[π₀.₅](https://arxiv.org/abs/2504.16054) | 论文 §4.1 统一 950 episode 训练集 |
| 遥操作栈 | Isaac Lab CloudXR + Apple Vision Pro | 腕位 IK + dex-retargeting 适配多手 |
| 同机构 loco-manip | [CoorDex（arXiv:2606.23680）](https://arxiv.org/abs/2606.23680) | UNC/Berkeley 人形 dexterous loco-manipulation 姊妹线 |

## 摘要级要点

- **问题：** 通才灵巧策略需要 benchmark 同时覆盖 **任务多样性**、**多臂–手具身**、**可控视觉变化** 与 **专家示范**；现有平台往往在 dexterous hand、visual variation、demo 数据或并行 RL 上缺项（Table 1）。
- **DexVerse 规模：** **100** 任务分 **8** 类（primitive / functional / articulation / non-prehensile / contact-rich / bimanual / multi-goal / long-horizon）；**3** 机械臂（Franka Research 3、UR10e、xArm 7）× **6** 灵巧手（Sharpa Wave、WUJI、Shadow、Inspire、Allegro、LEAP），各手另有 floating 变体。
- **模块化设计：** 任务 $\mathcal{T}=(\Omega,\mathcal{S}_0,\mathcal{O},\mathcal{A},\mathcal{G})$；场景/资产/观测/动作/初始化/成功谓词/随机化由配置类指定，基于 Isaac Lab manager-based 接口；具身通过紧凑 robot config 切换，无需重写任务逻辑。
- **视觉与非视觉变化：** 纹理、桌面材质、HDR 天空盒、光照、曝光、色温、相机视点；可与物体初始位姿、动力学参数等 **独立或联合** 启用。
- **资产：** PartNet-Mobility、ManiTwin、Isaac Lab/Sim、AutoBio、Synthesis；缺 mesh 时用 Meshy image-to-3D + 人工后处理。
- **数据集：** VR 遥操作 **3,180** 轨迹——56 单目标任务各 **55** 条（Shadow **50** + 其余 5 手各 **1**）；5 长时域任务各 **20** 条。存 **action–state** 对 + **state replay** 再生观测（跨机器物理漂移友好）。
- **观测模态：** 同步 proprio、RGB、depth、point cloud、simulator state。
- **Baseline 评测（19 任务 × 50 episode rollout）：** π₀.₅ 与 DP3 **并列最高 0.34**；DP **0.32**；OpenVLA **0.19**。无单一方法统治四类技能族；PushT / InsertPen / SlideUtilityKnife / OpenLaptop 等 **全线 0% 或近零**。
- **三条发现：** ① 互联网规模 VLA 预训练尚未转化为 dexterous 优势；② **最优观测模态依技能而异**（2D 够用于 pick-lift，点云利于 tool use，语言/flow 利于 articulation & precision）；③ **亚厘米对齐与持续力控** 仍是共同短板。
- **局限：** 当前聚焦 **仿真可复现 benchmark**；真机迁移、更多具身/任务族与标准化 cross-embodiment 评测留作未来工作。

## 核心摘录（面向 wiki 编译）

### 1) 任务分类（Table 2 摘要）

| 类别 | # | 代表任务 | 关键挑战 |
|------|---|----------|----------|
| Primitive | 9 | PickCube, StackCube, RelocateSphere | 简单目标、低动作复杂度 |
| Functional | 11 | HammerStrike, RetrieveCup, PourCan | 功能部位与 affordance |
| Articulation | 18 | OpenStapler, OpenLaptop, SqueezeScissors | 关节/部件约束运动 |
| Non-prehensile | 5 | PushT, TakeBook, PivotCuboid | 推/滑/枢轴/环境接触 |
| Contact-rich | 8 | InsertPeg, PlugCharger, NutThread | 精密对齐与持续接触 |
| Bimanual | 5 | BiLiftTray, BiHandover, BiLiftBox | 双手协调与传递 |
| Multi-goal | 39 | GraspMug + PushButton 等组合 | 多子目标组合 |
| Long-horizon | 5 | MakeCoffee, MicrowaveFood, CleanTable | 多阶段时序流程 |

### 2) 与代表性 benchmark 对比（Table 1 归纳）

| Benchmark | 灵巧手 | 多具身 | 视觉变化 | Demo | 并行 RL |
|-----------|--------|--------|----------|------|---------|
| CALVIN / LIBERO | ✗ | ✗ | △ | ✓ | ✗ |
| RoboTwin 2.0 | ✗ | ✓ | ✓ | ✓ | △ |
| ManiSkill3 | △ | △ | △ | △ | ✓ |
| DexMimicGen | ✓ | ✓ | ✗ | ✓ | ✗ |
| DexJoCo | ✓ | ✓ | ✓ | ✓ | ✗ |
| **DexVerse** | ✓ | ✓ | ✓ | ✓ | ✓ |

### 3) Baseline 在线成功率（Table 3 均值与亮点）

- **均值：** π₀.₅ **0.34** = DP3 **0.34** > DP **0.32** > OpenVLA **0.19**
- **Pick-and-Lift：** DP 最强（如 BimanualLiftCarton **0.94**）
- **Tool Use：** DP3 领先（FunctionalPourMug **0.64**）
- **Articulated：** π₀.₅ **0.35** 均值领先
- **Precision / 接触密集：** 全线崩溃（PushT **0.00** 全员）

### 4) 遥操作与 retargeting

- Apple Vision Pro → CloudXR → 腕位 IK 跟踪人手腕；指关节经 **optimization-based dex-retargeting** 映射到不同手 URDF。
- 新臂：更新 EE frame、初始姿态、低层控制器；新手：配置 keypoints、对应 link、retarget scale。

## 对 wiki 的映射

- [DexVerse](../../wiki/entities/paper-dexverse.md) — 问题定义、任务/具身/数据/评测归纳
- [Manipulation](../../wiki/tasks/manipulation.md) — 灵巧操作仿真 benchmark 代表工作
- [Isaac Lab](../../wiki/entities/isaac-lab.md) — 平台与 manager-based 环境接口
- [灵巧操作数据管线](../../wiki/queries/dexterous-manipulation-data-pipeline.md) — VR 示范 + 多模态观测 benchmark 对照
- [CoorDex](../../wiki/entities/paper-coordex-dexterous-humanoid-loco-manipulation.md) — 同 UNC/Berkeley 灵巧研究线交叉引用

## 引用（arXiv BibTeX）

```bibtex
@article{yao2026dexverse,
  title   = {DexVerse: A Modular Benchmark for Multi-Task, Multi-Embodiment Dexterous Manipulation},
  author  = {Yao, Yunchao and Xu, Zhuxiu and Zhang, Tianqi and Liu, Zixian and Li, Sikai and Wei, Zhenyu and Chen, Feng and Huang, Dihong and Wan, Kechang and Ma, Chenyang and Zhao, Shuqi and Gao, Shenghua and Tomizuka, Masayoshi and Ma, Yi and Ding, Mingyu},
  journal = {arXiv preprint arXiv:2607.08751},
  year    = {2026}
}
```
