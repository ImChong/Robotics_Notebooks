Warning: truncated output (original token count: 9325)
Total output lines: 343

---
title: Motion Retargeting（动作重定向）
type: concept
status: complete
created: 2026-04-14
updated: 2026-11-05
summary: 将人类或动物参考动作映射到异构机器人骨架上，在保留运动风格和语义的同时满足机器人的关节限制和动力学约束。
---

# Motion Retargeting（动作重定向）

## 是什么

Motion Retargeting 是将一个运动序列（通常来自人类或动物）**转换为适合目标机器人执行的动作序列**的过程。

核心挑战：源（人/动物）和目标（机器人）往往有不同的：
- 骨架拓扑（关节数量/连接方式）
- 肢体比例
- 关节限制
- 质量分布与动力学

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Retargeting | Motion Retargeting | 将人体/动物动作映射到目标机器人骨架 |
| MoCap | Motion Capture | 最常见参考动作来源 |
| IK | Inverse Kinematics | 满足末端/姿态约束的关节解算 |
| WBT | Whole-Body Tracking | 重定向后用于仿真跟踪训练 |
| AMP | Adversarial Motion Prior | 可与重定向数据组合约束运动风格 |

---

## 为什么重要

| 应用场景 | 重定向的作用 |
|---------|------------|
| 模仿学习参考轨迹 | 将 MoCap 数据转为机器人可执行轨迹，作为 RL 奖励或 BC 数据 |
| 全身遥操（Teleoperation） | 实时将人类动作映射到人形机器人 |
| AMP / ASE 风格先验 | 为 RL 策略提供运动风格参考 |
| 技能库建立 | 一次录制，多种机器人复用 |
| Ego → 灵巧 robot 动作 | [EgoWild2Dex](../entities/paper-egowild2dex.md) 用 **IK + 手指 retarget** 把 ego/glove/robot 示范统一到 **robot-native 高 DoF** 命令 |

---

## 主要方法

### 推理时具身感知反应选择

[Embodiment-Aware Inference-Time Search](../entities/paper-embodiment-aware-reaction-selection.md) 将冻结的人体反应生成器作为候选提议器：采样多个反应，经目标人形重定向后，以交互语义门限和机器人侧运动学可行性共同排序。Inter-X + G1 主设置中可行性提升 1.61%，但完整 K=16 路径约 9.2 Hz，论文未实机驱动；因此它补充的是**源反应选择层**，不是新的底层控制器。

### 工程工具：frax

[frax](../entities/frax.md) 的差分 IK 示例将 LAFAN1 动作重定向到 G1，提供关节与地面约束及可选自碰撞处理。

### 1. 基于关节角度映射（Joint-Space Retargeting）
最简单的方法：对每个关节直接建立角度映射（scale + offset）：
```python
θ_robot[i] = scale[i] * θ_human[i] + offset[i]
```
**优点**：实时，无优化；**缺点**：不考虑运动学差异，可能末端位置偏差大

### 2. 基于任务空间的优化（Task-Space IK）
以末端执行器（手、脚）的目标位置为约束，求解机器人的关节角：
```
minimize  ‖θ - θ_ref‖²
subject to: FK(θ) = p_target (末端位置约束)
            θ_min ≤ θ ≤ θ_max
            接触约束
```
工具：Pinocchio + TSID / Crocoddyl；模块化 JAX 栈见 [PyRoki](../entities/paper-notebook-pyroki.md)（手/人形 retarget 示例 + [ProtoMotions](../entities/protomotions.md) v3 默认后端）

### 3. 基于物理的重定向（Physics-Based Retargeting）
用物理仿真器验证重定向后的动作是否可执行（不摔倒）：
- 先在仿真中播放参考动作，用 PD 控制器追踪
- 收集可行段，过滤摔倒片段
- 可结合 RL 做后续精修（AMP 风格）
- **双层 RL 式（ReActor）**：上层优化**参数化运动学参考**（稀疏语义刚体对应 + 有界偏移），下层用 **RL 跟踪**并在仿真里回传误差；把「造参考」与「跟参考」联立，减少脚滑与自碰等运动学伪影，详见 [ReActor](../methods/reactor-physics-aware-motion-retargeting.md)。
- **采样式物理重定向（SPIDER）**：在**并行物理仿真**中对控制序列做**采样优化**（退火噪声核），把人体+物体的**运动学参考** refinement 成**动力学可行**轨迹；用**课程式虚拟接触力**稳定接触丰富任务中的序列歧义，详见 [SPIDER](../methods/spider-physics-informed-dexterous-retargeting.md)。
- **增量 SBTO（DynaRetarget）**：用 **CEM + MuJoCo rollout** 对 PD 目标 knot 做 **incremental full-horizon** 采样轨迹优化，把 IK/kinematic 参考 refinement 为长时域 loco-manipulation 可行轨迹，相对 SBMPC 基线成功率约翻倍，详见 [DynaRetarget / SBTO](../methods/dynaretarget-sbto-motion-retargeting.md).
- **接触隐式多重打靶（DSMS / Shooting for Contact）**：把可微仿真器离散转移嵌进 **IPOPT 多重打靶 NLP**，接触/摩擦/冲击在仿真内解析、无需接触时刻表；支持任意路径约束与命令条件化步态库，加速下游 motion-imitation RL，并在 G1 上零样本爬行 / 180° 跳转，详见 [DSMS](../methods/dsms-contact-implicit-multiple-shooting.md) 与 [Shooting for Contact](../entities/paper-shooting-for-contact.md)。
- **接触感知运动学精炼（CoRe）**：在 RL 之前用趾轨迹检测接触段，再做接触约束轨迹优化、足偏航与自碰平滑；主张「先修参考、再跟踪」。Humanoids 2025 论文含接触奖励 RL；开源 [CoRe v0.1.0](../entities/paper-core.md) 兑现重定向+精炼（Kimodo/GEM-X → 11 机），T2M/RL 未随仓发布。前端跨骨架统一见姊妹 [RMR](../entities/paper-rmr.md)。勿与 [PhysCoRe](../entities/paper-physcore.md) 混淆。
- **GRF 锚定多接触 TO（KDMR）**：用同步 **地面反力** 推断 heel–toe 接触日程，再以 CasADi+Pinocchio 全身 NLP 强制动力学与无滑约束，减轻 GMR 类脚浮空伪影并加速 BeyondMimic 跟踪；详见 [KDMR](../entities/paper-kdmr.md)。
- **骨架 URDF 校准 + 渐进 KDTO（SPARK）**：先校准 human URDF 到目标人形再 IK，再经 KTO→ID→KDTO 产出动力学可行轨迹与力矩参考，服务 side flip 等高动态；详见 [SPARK（骨架对齐重定向）](../entities/paper-spark-skeleton-aligned-retargeting.md)。
- **交互保留灵巧重定向（TopoRetarget）**：在 **hand–object interaction mesh** 上匹配 **距离加权 Laplacian 坐标** + 骨方向先验与穿透约束，把人手演示转为灵巧手可学的接触保真参考（~5 ms/帧）；下游轻量 PPO 残差跟踪可在 Pen-Spin 等任务上显著优于 OmniRetarget 等基线，并零样本部署 [Wuji Hand](../entities/wuji-robotics.md)；[官方代码](https://github.com/wuji-technology/toporetarget)（MIT）已开源，详见 [TopoRetarget](../methods/toporetarget-interaction-preserving-dexterous-retargeting.md)。

### 3.5 稀疏关键点重定向（SKR，BifrostUMI）

[BifrostUMI](../entities/paper-bifrost-umi.md) 提出的 **Spatial Keypoint Retargeting（SKR）** 面向 **无机器人采集 → 人形部署**：用 **骨盆、左右 TCP、左右脚** 五个任务关键点表示全身运动，**仅** 按身高差缩放 **骨盆–脚垂直距离**，其余关键点间 **度量空间关系保持不变**（对比 [GMR](../methods/motion-retargeting-gmr.md) 的全局/局部缩放）。闭环中从关节 FK 得当前关键点，与扩散高层预测合成目标，再用 **mink** 解全身 IK 供 WBC 跟踪——把「几何桥」与「低层动力学」明确分层。

### 3.6 采样式灵巧手重定向（SBR，Smooth Operator）

[mimic robotics](https://mimicrobotics.github.io/smooth-operator/) 的 **Sampling-Based Retargeter（SBR）** 面向 **15 DoF 级实时遥操作**：用 **Kabsch–Umeyama** 对齐人手指点云与机器人手，再以 **MPPI / MPOPI + iCEM** 做 **梯度无关** 路径积分优化，缓解 DexPilot / GeoRT 等 **梯度法局部极小与抖动**。18 人用户研究中整体成功率 **54.1%**、NASA-TLX **36.4**（相对 GeoRT 26.6% / 56.4）。开源快照见 [mimic_retargeter_lab](https://github.com/mimicrobotics/mimic_retargeter_lab)；与 [mimic wearable U1](../entities/mimic-wearable-u1.md)「机械 1:1、无软件重定向」形成 **中层采集 vs 顶层真机遥操作** 对照。

### 3.7 学习式稠密点云对应（UMR）

[UMR](../entities/paper-umr-unified-motion-retargeting.md)（arXiv:2609.02134）在规范 T-pose 上学人–机外表面点对，再作为约束优化锚点匹配位置、法向与接触向量，**不手写骨架关键点表**。换源（SMPL-X / SOMA / 扫描网格）与换机主要复用同一套索引。跟踪对照 [GMR](../methods/motion-retargeting-gmr.md)，接触对照 [OmniRetarget](../entities/paper-hrl-stack-03-omniretarget.md)；[官方 GitHub](https://github.com/hanyang9/UMR) 与 [UMR Studio](https://hanyang9.github.io/UMR/umr_studio.html) 已开源（2026-09-12）。

### 3.7 非拟人三指夹爪遥操作重定向（VTAP）

[VTAP Gripper](../entities/paper-vtap-gripper.md)（arXiv:2607.15448）面向 **异构三指夹爪**（无人手关节一一对应）：用 **cage/power/pinch 手势锁子空间** + 中间坐标系 \(\mathcal{I}\)，再优化指尖位置/朝向；singulation 另把拇指–食指滚动映射到内收/外展。这是 **末端夹爪遥操作** 重定向，不是全身 MoCap→人形 WBT 主线，但与 SBR/TopoRetarget 同属「人手 → 非同构手」问题族。

### 3.8 离线全身闭式重定向（WARP）

[WARP](../entities/paper-warp-whole-body-retargeting.md)（arXiv:2606.29940，Georgia Tech）面向 **无机器人在环的 Meta Quest 离线人演示 → 全身移动操作 BC**：核心 **c-SEW** 以 **palm 硬约束 + adaptive offset + Stereo-sew/SP3** 在 SEW 表示上闭式求解，避免 MINK 类加权 IK 的 **不精确**（EEF vs 全身 trade-off）与 **不一致**（冗余多解 → action multi-modality）；**lazy mobile-base** 让 6-DoF torso 吸收微调。离线设定无遥操作闭环纠错，重定向轨迹即监督——项目页报告 palm 误差相对 MINK-EF 降 **>150×**，DexMimicGen 策略平均成功率 **71% vs 59%**（replay 相近）。截至入库日 **未开源**。

### 3.9 硬件共设计可以省略软件重定向

当采数装置与目标手在 **运动学、接触几何、外观与时序** 上做成同构「双胞胎」时，手指状态可以直接进机器人关节空间，**不再经过** 人手→异构手的优化/学习式 retarget。产业样本：[mimic U1](../entities/mimic-wearable-u1.md) 刚性约束到 M1；[TwinDEX](../entities/twindex.md)（自变量，2026-09）三指外骨骼–机械手五维对齐，并主张纯 robot-free 数据即可部署。这不是「重定向做对了」，而是 **用机械约束把问题定义掉**——代价是 embodiment 锁定，且两例截至入库日均 **未开源硬件/训练栈**。需要跨臂复用时仍走 [HandUMI](../entities/handumi.md) / UMI 族的标定 + 重定向。

### 4. 深度学习重定向（Learning-Based）
- Encoder-Decoder 架构：将人类骨架 embedding，再 decode 到目标机器人
- 可跨模态（视频 → 机器人关节）
- 近年的工程趋势是先做几何重定向，再接一个下游物理一致化或 tracking 层，避免“姿态像但动力学不可执行”
- **代表工作**：[NMR（神经运动重定向与人形全身控制）](../methods/neural-motion-retargeting-nmr.md) 用仿真 RL 专家构造物理一致的人机配对数据，再训练 CNN–Transformer 做整段 SMPL→机器人映射，显式针对优化式重定向的非凸与噪声传播问题。

---

---

## 重定向不等于控制策略：两层架构模式

在实际的人形机器人工程中，**重定向只是起点**。单纯的几何映射往往无法直接上机。

### 1. 运动学重定向层 (Kinematic Layer)
- **代表方法**：[GMR (General Motion Retargeting)](../methods/motion-retargeting-gmr.md)。
- **作用**：解决姿态、角度、关键点坐标的映射。
- **局限**：不能保证质心平衡、加速度连续性、接触力可行性以及力矩安全。容易出现脚部滑动或自碰撞。
- **动画…3325 tokens truncated…2607_15448.md) — VTAP：非拟人三指夹爪手势条件遥操作重定向（arXiv:2607.15448）
- **ingest 档案：** [sources/papers/bifrost_umi_arxiv_2605_03452.md](../../sources/papers/bifrost_umi_arxiv_2605_03452.md) — BifrostUMI SKR：稀疏关键点 + 仅身高缩放 + mink IK（arXiv:2605.03452）
- **ingest 档案：** [sources/papers/warp_arxiv_2606_29940.md](../../sources/papers/warp_arxiv_2606_29940.md) — WARP：离线 Meta Quest 全身演示的闭式 c-SEW 重定向 + lazy base（arXiv:2606.29940）；配套 [项目页](../../sources/sites/warp-retargeting-github-io.md)（截至入库日未列代码）
- **ingest 档案：** [sources/sites/jc-bao-spider-project-github-io.md](../../sources/sites/jc-bao-spider-project-github-io.md) — SPIDER 项目页 jc-bao.github.io/spider-project（管线演示与 BibTeX）
- **ingest 档案：** [sources/sites/amass-dataset.md](../../sources/sites/amass-dataset.md) — AMASS：SMPL 统一人体动捕元数据集（MPI-IS 站点与 ICCV 2019 论文索引）
- **ingest 档案：** [sources/repos/ubisoft-laforge-animation-dataset.md](../../sources/repos/ubisoft-laforge-animation-dataset.md) — LaFAN1：Ubisoft BVH 动捕与评估脚本（SIGGRAPH 2020 论文配套）
- **ingest 档案：** [sources/repos/omomo_release.md](../../sources/repos/omomo_release.md) — OMOMO：人–物交互 MoCap（SIGGRAPH Asia 2023）
- **ingest 档案：** [sources/repos/phuma.md](../../sources/repos/phuma.md) — PHUMA：G1/H1-2 预重定向 locomotion（arXiv:2510.26236）
- **ingest 档案：** [sources/sites/humanoideveryday.md](../../sources/sites/humanoideveryday.md) — Humanoid Everyday：真机开放世界操作集
- **ingest 档案：** [sources/sites/mixamo.md](../../sources/sites/mixamo.md) — Mixamo：Adobe 在线角色与动画服务（商业许可与管线说明）
- **ingest 档案：** [sources/papers/egohtr_arxiv_2607_13472.md](../../sources/papers/egohtr_arxiv_2607_13472.md) — EgoHTR：rough-terrain 场景对齐人演示；Human2Robot 侧用 OmniRetarget/GMR/CoACD
- **ingest 档案：** [sources/papers/kdmr_arxiv_2603_09956.md](../../sources/papers/kdmr_arxiv_2603_09956.md) — KDMR：GRF 锚定多接触动力学重定向（arXiv:2603.09956）
- **ingest 档案：** [sources/papers/spark_skeleton_aligned_retargeting_arxiv_2603_11480.md](../../sources/papers/spark_skeleton_aligned_retargeting_arxiv_2603_11480.md) — SPARK：URDF 校准 + 渐进 KDTO（arXiv:2603.11480）
- **ingest 档案：** [sources/papers/umr_unified_motion_retargeting_arxiv_2609_02134.md](../../sources/papers/umr_unified_motion_retargeting_arxiv_2609_02134.md) — UMR：学习点云对应的统一重定向（arXiv:2609.02134）
- **ingest 档案：** [sources/repos/core_retarget.md](../../sources/repos/core_retarget.md) — CoRe v0.1.0：SOMA 接触感知重定向（11 机，Apache-2.0）；论文见 [core_humanoids_2025.md](../../sources/papers/core_humanoids_2025.md)、[rmr_iros_2025.md](../../sources/papers/rmr_iros_2025.md)

---

## 关联页面
- [EmbodyHOI](../entities/paper-embodyhoi.md) — 在冻结 HOI 生成器采样中嵌入可微 IK，替代事后 retargeting
- [FlashDexRetarget](../entities/paper-flashdexretarget.md) — 一个策略联合训练多段灵巧手—物体演示；以离策略 RL 生成物理可行的重定向轨迹
- [具身数据采集四层术语地图](./embodied-data-collection-four-layers-taxonomy.md) — MoCap 教法层与重定向在动作标签形成中的位置
- [Motion Retargeting Pipeline](./motion-retargeting-pipeline.md) — 端到端工程链路视角：源归一 → 骨架对齐 → IK → 物理筛选 → 配对监督
- [KDMR](../entities/paper-kdmr.md) — GRF 多接触全身 TO
- [SPARK（骨架对齐重定向）](../entities/paper-spark-skeleton-aligned-retargeting.md) — URDF 校准 + KDTO
- [UMR（学习点云对应）](../entities/paper-umr-unified-motion-retargeting.md) — 稠密表面对应 + 接触图直传；不手写关键点（arXiv:2609.02134；[官方 GitHub](https://github.com/hanyang9/UMR) 已开源）
- [Motion Retargeting Objective（重定向目标函数形式化）](../formalizations/motion-retargeting-objective.md) — 姿态相似 / 末端接触 / 平衡 / 限位 / 平滑项的统一加权和及其三种工程退化
- [Motion Data Quality（动作数据质量维度）](./motion-data-quality.md) — 形态差距/接触/物理/规模四轴决定重定向是否可省略及需补几层
- [Teleopit](../entities/paper-teleopit.md) — 归一化指方向 + 距离/拇指帧的跨形态灵巧手在线优化重定向（somehand；arXiv:2608.01834）
- [EgoHTR](../entities/paper-egohtr.md) — 上游 rough-terrain 4D 人–场景数据（待开放）与 G1 retarget 叙事
- [人形训练数据管线选型指南](../queries/humanoid-training-data-pipeline.md) — 重定向在「来源 → 重定向 → 训练范式」端到端管线中的位置
- [Chasing Autonomy Pipeline](../methods/chasing-autonomy-pipeline.md) — 利用带硬约束的动态重定向来生成改进参考库的框架
- [Imitation Learning](../methods/imitation-learning.md) — 重定向后的轨迹常作为模仿学习的参考数据
- [Locomotion](../tasks/locomotion.md) — locomotion 的风格先验来自重定向后的 MoCap 数据
- [Loco-Manipulation](../tasks/loco-manipulation.md) — 全身操作任务需要手臂 + 腿部的联合重定向
- [Whole-Body Control](./whole-body-control.md) — WBC 执行重定向后的参考轨迹
- [Whole-Body Tracking Pipeline](./whole-body-tracking-pipeline.md) — 「映射 → 训练 → 迁移」三段流水线的中段：把重定向产物当作训练数据学跟踪策略
- [跨具身策略迁移选型指南](../queries/cross-embodiment-transfer-strategy.md) — 三段流水线的末段：已训策略/数据搬到新机体的选型决策树
- [SONIC vs BeyondMimic vs SD-AMP vs Heracles](../comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md) — 四条主流 WBT 方法谱系对比（重定向下游的训练/迁移取舍）
- [Sim2Real](./sim2real.md) — 重定向数据质量影响真实机器人策略的泛化性
- [GMR (通用动作重定向)](../methods/motion-retargeting-gmr.md) — 基于运动学优化的重定向代表实现
- [NMR（神经运动重定向与人形全身控制）](../methods/neural-motion-retargeting-nmr.md) — 学习式整段映射 + 仿真 RL 修补监督
- [ReActor（物理感知 RL 运动重定向）](../methods/reactor-physics-aware-motion-retargeting.md) — 双层：可学习参考 + 同环 RL 跟踪，近似上层梯度
- [SPIDER（物理感知采样式灵巧重定向）](../methods/spider-physics-informed-dexterous-retargeting.md) — 运动学参考 + 并行仿真采样优化 + 课程式虚拟接触引导
- [TopoRetarget（交互保留灵巧重定向）](../methods/toporetarget-interaction-preserving-dexterous-retargeting.md) — hand–object interaction mesh + 实时 Laplacian 优化，面向灵巧 in-hand manipulation 参考生成
- [VTAP Gripper](../entities/paper-vtap-gripper.md) — 非拟人三指夹爪的手势条件遥操作重定向（硬件/采集侧）
- [TwinDEX](../entities/twindex.md) — 三指外骨骼–同构手共设计，用硬件对齐 **省略** 软件 retarget
- [GMR vs NMR vs ReActor（重定向方法谱系对比）](../comparisons/gmr-vs-nmr-vs-reactor.md) — 三条主流路线在误差修补位置、训练/推理成本、跨形态能力上的选型坐标
- [ExoActor](../methods/exoactor.md) — 视频生成驱动的人形控制流水线，提供"中间重定向并非永远收益项"的反例
- [Character Animation vs Robotics](./character-animation-vs-robotics.md) — 当目标函数里同时出现表演意图与物理可控性时的张力与案例切片
- [MotionCode](../entities/motioncode.md) — 商业运动数据与「人形/具身 + RL」叙事样本
- [FreeMoCap](../entities/freemocap.md) — 低成本开源动捕软件栈，与重定向 / 仿真训练组合使用时的入口之一
- [AMASS](../entities/amass.md) — SMPL 系大规模统一动捕档案，常与 AMP / 生成式运动模型一起出现
- [LaFAN1](../entities/lafan1-dataset.md) — BVH 多主题棚拍数据与过渡任务基准（注意 NC-ND 许可）
- [OMOMO](../entities/omomo-dataset.md) — 人–物交互 MoCap，OmniRetarget `robot-object/` 源
- [PHUMA](../entities/dataset-bfm-phuma.md) — 预重定向 G1/H1-2 locomotion（PhySINK）
- [Humanoid Everyday](../entities/humanoid-everyday-dataset.md) — 真机人形操作多模态集
- [人形数据五集选型](../comparisons/humanoid-reference-motion-datasets.md)
- [MoReFlow（flow matching 无配对重定向）](../entities/paper-moreflow-motion-retargeting-flow.md) — VQ-VAE token + flow matching 跨角色对齐（arXiv:2509.25600）
- [AdaMorph（统一 Transformer 重定向）](../entities/paper-adamorph-unified-motion-retargeting.md) — 12 机种一模型；intent + AdaLN（arXiv:2601.07284；≠ UMR 点云对应）
- [三路技术地图 Query](../queries/motion-retargeting-three-routes-landscape.md) — 深蓝 2026-09-07 公众号综述索引
- [OmniRetarget](../entities/paper-hrl-stack-03-omniretarget.md) — interaction mesh 硬约束 + 单演示增广的交互保留重定向（ICRA 2026）
- [holosoma](../entities/holosoma.md) — OmniRetarget 开源实现与 WBT/locomotion 训练部署框架
- [OmniRetarget 数据集](../entities/omniretarget-dataset.md) — G1 交互重定向轨迹 HuggingFace 公开子集
- [Mixamo](../entities/mixamo.md) — 商业动画库，与科研向 MoCap 档案对照阅读
- [SAM 3D Body](../entities/sam-3d-body.md) — 单目 RGB 全身 MHR 估计，可作视频→重定向上游
- [SAM3DBody-cpp](../entities/sam3dbody-cpp.md) — 上述模型的 C++/BVH 工程导出
- [BifrostUMI（论文实体）](../entities/paper-bifrost-umi.md) — Robot-Free 示范的 SKR 与全身 WBC 接口
- [WARP（论文实体）](../entities/paper-warp-whole-body-retargeting.md) — 离线人演示闭式全身重定向；开环回放 + BC 全身移动操作（arXiv:2606.29940）
- [PHC](../entities/phc.md) — SMPL fitting 重定向与大规模物理模仿
- [SOMA Retargeter](../entities/soma-retargeter.md) — SOMA/SEED→G1 批处理重定向
- [robot_retargeter](../entities/robot-retargeter.md) — SMPL-X / LAFAN1 CSV→多机型 mink IK 重定向
- [CoRe v0.1.0](../entities/paper-core.md) — SOMA Kimodo/GEM-X → 11 机接触精炼；论文 [CoRe](../entities/paper-core.md) / [RMR](../entities/paper-rmr.md)
- [GVHMR](../entities/gvhmr.md) / [VideoMimic](../entities/videomimic.md) — 视频→人体/人形两条管线入口
- [human2humanoid](../entities/human2humanoid.md) — 遥操与 AMASS 重定向同仓
- [mocap_retarget](../entities/mocap-retarget.md) — 轻量工程向 MoCap 映射参考
- [motion_imitation（四足）](../entities/motion-imitation-quadruped.md) / [STMR](../entities/stmr-quadruped-retargeting.md) / [AMP_for_hardware](../entities/amp-for-hardware.md) / [MetalHead](../entities/metalhead.md) / [LeggedGym-Ex](../entities/leggedgym-ex.md) — 四足模仿与 AMP 生态
- [Go2 Motion Imitation](../entities/go2-motion-imitation.md) — Go2 + Genesis 重定向样板
- [Kine2Go 数据集](../entities/paper-kine2go.md) — 40 段多来源参考动作经 Go2 重定向与 PPO 模仿，发布 800 条仿真 rollout。
- [PAN Motion Retargeting](../entities/pan-motion-retargeting.md) / [WalkTheDog](../entities/walk-the-dog.md) — 跨形态（人↔四足）学习式对齐研究
- [AMP-RSL-RL](../entities/amp-rsl-rl.md) — rsl_rl(PPO)+AMP 的人形模仿实现，重定向/动捕产物的消费侧（可 pip 安装）
- [fairmotion](../entities/fairmotion.md) — 上游通用动捕数据处理库（已归档），重定向前的数据 IO/表示层
- [KinoGen](../entities/paper-kinogen.md) — 从生成视频拟合机器人与物体模型，绕过人体到机器人的逐帧重定向

- 相关研究：[flowhmr](../entities/paper-flowhmr.md)。
