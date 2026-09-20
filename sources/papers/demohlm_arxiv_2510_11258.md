# DemoHLM: From One Demonstration to Generalizable Humanoid Loco-Manipulation（arXiv:2510.11258）

> 来源归档（ingest）

- **标题：** DemoHLM: From One Demonstration to Generalizable Humanoid Loco-Manipulation
- **类型：** paper / humanoid / loco-manipulation / imitation-learning / data-generation / mimicgen / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2510.11258>
- **arXiv HTML：** <https://arxiv.org/html/2510.11258v1>
- **PDF：** <https://arxiv.org/pdf/2510.11258>
- **项目页：** <https://beingbeyond.github.io/DemoHLM/>
- **代码：** <https://github.com/BeingBeyond/DemoHLM>（截至 2026-09-20 仅 README + 项目站镜像，**无可运行训练/推理脚本**）
- **机构：** 北京大学（PKU）、超越智能（BeingBeyond）；通讯作者 Zongqing Lu
- **作者：** Yuhui Fu*、Feiyang Xie*、Chaoyi Xu、Jing Xiong、Haoqi Yuan、Zongqing Lu §（* 共一）
- **发表：** arXiv 预印本 2025-10-11；IEEE RA-L 2026-02-19（DOI: 10.1109/LRA.2026.3666395）
- **硬件：** Unitree G1 + 2-DoF 主动颈 + Intel RealSense D435 RGB-D；部分任务换 3D 打印平行夹爪
- **入库日期：** 2026-09-20
- **一句话说明：** 仿真中 VR 遥操作采集 **单条示范** → 物体/本体坐标系分段重放 + AMO 式 RL 全身控制器 → 合成数百–数千条轨迹 → BC（ACT/DP/MLP）训练高层操纵策略；G1 真机 **10 任务** 空间泛化 zero-shot 部署。

## 摘要级要点

- **问题：** 人形 loco-manipulation 常依赖任务定制 RL 奖励、大量真机遥操作数据，或 SMPL 仿真方案难以上真机；MimicGen 系工作限于固定底座机械臂。
- **层次架构：** 低层 **AMO 式 RL 全身控制器**（50 Hz）将 $(v_x,v_y,\omega,h,r,p,y,\mathbf{q}_{upper})$ 映射为全身关节 PD 目标；高层 **BC 操纵策略**（10 Hz）闭环输出上述高层命令，观测含本体 + 相机系物体 6D 位姿。
- **数据生成（MimicGen 扩展）：** 单条 VR 示范 $\tau^h$ 在首次接触 $t_c$ 切为 pre-contact / post-contact；pre 段 **object-centric** 末端位姿，post 段 **proprioception-centric**；三阶段拼接：**locomotion**（PD 速度引导靠近）→ **pre-manipulation**（插值对齐 + object-centric 重放）→ **manipulation**（切换 proprio-centric 处理「相对物体静止」段）。
- **示范采集：** Apple Vision Pro + VisionProTeleop → Pink IK 解 $(h,r,p,y,\mathbf{q}_{upper})$；仿真 IsaacGym + SAPIEN/PartNet-Mobility 资产。
- **BC 架构：** 对比 ACT、带 action chunk 的 MLP、Diffusion Policy；5k 合成数据规模下 ACT/DP 显著优于 MLP。
- **真机感知：** 单头 RealSense D435 + **FoundationPose++** 估计/跟踪物体 6D 位姿（与仿真观测对齐）；Unitree SDK v2 PD @ 500 Hz。

## 核心摘录（面向 wiki 编译）

### 十项 loco-manipulation 任务

| 分组 | 任务 | 要点 |
|------|------|------|
| 橡胶手 | LiftBox / PressCube / PushCube / Handover | 推/按/抬/双手交接 |
| 平行夹爪 | GraspCube / OpenCabinet / PushCart / EraseBoard / PourWater / ExchangeCube | 抓取/拉柜/推车/擦板/倒水/换手 |

### 仿真成功率 vs 数据规模（Table 1，5k 轨迹，5000 rollout × 3 seed）

| 任务 | 100 | 500 | 1k | 5k |
|------|-----|-----|-----|-----|
| LiftBox | 86.4 | 96.2 | 98.6 | **98.8** |
| PushCart | 75.8 | 92.6 | 93.4 | **95.8** |
| GraspCube | 57.0 | 75.5 | 77.6 | **87.9** |
| OpenCabinet | 18.9 | 49.5 | 54.9 | **67.3** |
| Handover | 28.5 | 47.0 | 52.1 | **57.5** |
| ExchangeCube | 13.4 | 34.0 | 44.3 | **52.9** |

趋势：**合成数据量与策略成功率单调正相关**，边际收益递减。

### BC 架构（Table 2，5k 数据）

- ACT 与 Diffusion Policy 整体接近且优于 MLP；MLP 在 Handover / EraseBoard / ExchangeCube 等长时序/双手任务跌幅最大。
- 论文默认报告 ACT 结果。

### 真机 zero-shot（Table 4，每任务 5 trials）

| LiftBox | PressCube | PushCube | Handover | GraspCube | OpenCabinet | EraseBoard |
|---------|-----------|----------|----------|-----------|-------------|------------|
| 5/5 | 5/5 | 4/5 | 4/5 | 3/5 | 2/5 | 2/5 |

PushCart / PourWater / ExchangeCube 等未列入该表（论文 §4.4 侧重已部署的 7 项）。

### 局限（§5）

- 纯仿真数据 → 动力学/视觉 sim2real 间隙；单 RGB-D 在遮挡/ clutter 受限；依赖 FoundationPose 需物体模型，难泛化未建模物体。

## 对 wiki 的映射

- 沉淀实体页：[DemoHLM（161 #136 / arXiv）](../../wiki/entities/paper-loco-manip-161-136-demohlm.md)
- 交叉：[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[HumanoidMimicGen](../../wiki/entities/paper-humanoidmimicgen.md)、[Being-0](../../wiki/entities/paper-loco-manip-161-057-being-0.md)、[unitree-g1](../../wiki/entities/unitree-g1.md)

## 参考来源（原始）

- arXiv:2510.11258 — 论文正文
- [loco_manip_161_survey_136_demohlm.md](loco_manip_161_survey_136_demohlm.md) — 161 篇策展索引
- [demohlm 项目页归档](../sites/demohlm.md)
- [demohlm 代码归档](../repos/demohlm.md)
