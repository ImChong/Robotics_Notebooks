---
title: Motion Retargeting（动作重定向）
type: concept
status: complete
created: 2026-04-14
updated: 2026-06-22
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

---

## 主要方法

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
工具：Pinocchio + TSID / Crocoddyl

### 3. 基于物理的重定向（Physics-Based Retargeting）
用物理仿真器验证重定向后的动作是否可执行（不摔倒）：
- 先在仿真中播放参考动作，用 PD 控制器追踪
- 收集可行段，过滤摔倒片段
- 可结合 RL 做后续精修（AMP 风格）
- **双层 RL 式（ReActor）**：上层优化**参数化运动学参考**（稀疏语义刚体对应 + 有界偏移），下层用 **RL 跟踪**并在仿真里回传误差；把「造参考」与「跟参考」联立，减少脚滑与自碰等运动学伪影，详见 [ReActor](../methods/reactor-physics-aware-motion-retargeting.md)。
- **采样式物理重定向（SPIDER）**：在**并行物理仿真**中对控制序列做**采样优化**（退火噪声核），把人体+物体的**运动学参考** refinement 成**动力学可行**轨迹；用**课程式虚拟接触力**稳定接触丰富任务中的序列歧义，详见 [SPIDER](../methods/spider-physics-informed-dexterous-retargeting.md)。
- **增量 SBTO（DynaRetarget）**：用 **CEM + MuJoCo rollout** 对 PD 目标 knot 做 **incremental full-horizon** 采样轨迹优化，把 IK/kinematic 参考 refinement 为长时域 loco-manipulation 可行轨迹，相对 SBMPC 基线成功率约翻倍，详见 [DynaRetarget / SBTO](../methods/dynaretarget-sbto-motion-retargeting.md).
- **交互保留灵巧重定向（TopoRetarget）**：在 **hand–object interaction mesh** 上匹配 **距离加权 Laplacian 坐标** + 骨方向先验与穿透约束，把人手演示转为灵巧手可学的接触保真参考（~5 ms/帧）；下游轻量 PPO 残差跟踪可在 Pen-Spin 等任务上显著优于 OmniRetarget 等基线，并零样本部署 [Wuji Hand](../entities/wuji-robotics.md)，详见 [TopoRetarget](../methods/toporetarget-interaction-preserving-dexterous-retargeting.md)。

### 3.5 稀疏关键点重定向（SKR，BifrostUMI）

[BifrostUMI](../entities/paper-bifrost-umi.md) 提出的 **Spatial Keypoint Retargeting（SKR）** 面向 **无机器人采集 → 人形部署**：用 **骨盆、左右 TCP、左右脚** 五个任务关键点表示全身运动，**仅** 按身高差缩放 **骨盆–脚垂直距离**，其余关键点间 **度量空间关系保持不变**（对比 [GMR](../methods/motion-retargeting-gmr.md) 的全局/局部缩放）。闭环中从关节 FK 得当前关键点，与扩散高层预测合成目标，再用 **mink** 解全身 IK 供 WBC 跟踪——把「几何桥」与「低层动力学」明确分层。

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

### 2. 动力学一致化层 (Dynamical Layer)
- **作用**：在重定向轨迹的基础上，补足物理约束。
- **实现手段**：
    - **QP 优化 (如 HALO)**：通过约束二次规划，强制满足固定脚位置和地表接触约束，修正 Base 位姿漂移。
    - **RL Fine-tuning**：以重定向轨迹作为参考，通过 RL 在仿真中进行鲁棒性训练。
    - **WBC 跟踪**：将重定向轨迹作为 WBC 的任务目标，由底层控制器实时处理平衡与力矩限制。

---

## 上游衔接：数据来源 → 质量评估 → 重定向 → 策略输入

重定向不是链路的起点。它的**输入由前两段决定，输出喂给第四段**——是否需要重定向、需要补几层，并非由重定向本身决定，而是由**第②段的数据质量体检**给出判据：

| 段 | 视角 | 关键判据 | 主页面 |
|----|------|---------|--------|
| ① 数据来源 | 选 MoCap / 视频 / 真机执行数据 | 表示、规模、许可证、是否预重定向 | [人形数据五集选型](../comparisons/humanoid-reference-motion-datasets.md) |
| ② 质量评估 | 物理可行性 / 接触一致性 / 形态差距 / 规模多样性 四轴 | **形态差距大 → 本页重定向不可省略** | [Motion Data Quality](./motion-data-quality.md) |
| ③ **重定向（本页）** | 几何映射 + 动力学一致化，把人体参考变成机器人可执行参考 | 接触保真、限位满足、物理可行 | 本页 §「两层架构模式」 |
| ④ 策略输入 | 物理可行参考 → WBT / AMP / IL 训练数据 | 真机可执行的策略 | [人形训练数据管线选型指南](../queries/humanoid-training-data-pipeline.md) |

> 因果方向：[第②段四质量轴](./motion-data-quality.md) 中的**形态差距**轴直接决定本页是否被触发（差距可忽略时如真机执行数据 [Humanoid Everyday](../entities/humanoid-everyday-dataset.md) 可跳过重定向直接 IL）；**接触一致性 / 物理可行性**两轴则决定本页 §「动力学一致化层」要补 QP / RL 几层。整条链的端到端决策树见 [人形训练数据管线选型指南](../queries/humanoid-training-data-pipeline.md)。

---

## 三段流水线衔接：重定向产物 → WBT 训练数据 → 跨具身策略蒸馏

重定向常被误当成「终点」，但在人形动作落地的整条链里它只是**第一段**。把视角拉远，会看到「**映射 → 训练 → 迁移**」三段彼此咬合：

| 阶段 | 视角 | 产物 | 主页面 |
|------|------|------|--------|
| ① 映射（Mapping） | 几何/动力学一致化：把人体参考变成「机器人物理上可执行的参考」 | 物理可行参考轨迹 | [Motion Retargeting Pipeline](./motion-retargeting-pipeline.md)、[重定向目标函数形式化](../formalizations/motion-retargeting-objective.md) |
| ② 训练（Training） | 把重定向**产物当作训练数据**，学一个能稳定执行它们的全身跟踪策略 | 真机可执行的 WBT 策略 | [Whole-Body Tracking Pipeline](./whole-body-tracking-pipeline.md) |
| ③ 迁移（Transfer） | 把已训策略/数据**搬到新机体**，区分重训迁移 vs 高效后训练 vs 联合训练 | 跨具身复用的策略族 | [跨具身策略迁移选型指南](../queries/cross-embodiment-transfer-strategy.md)、[SONIC vs BeyondMimic vs SD-AMP vs Heracles](../comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md) |

衔接关系：本页（重定向）的 §「两层架构模式」产出**阶段 ①** 的物理可行参考；[WBT 流水线](./whole-body-tracking-pipeline.md) 把这些参考当**阶段 ②** 的训练数据消费，产出真机策略；当目标换成新机体时进入**阶段 ③**——重定向的「**仅缩放骨盆–脚距、其余度量关系保持**」（如 SKR）等几何桥接质量，会直接决定跨具身迁移能否复用同一份参考库。三段构成「**MoCap → Reference → Policy → 新机体**」的闭环；[Sim2Real](./sim2real.md) 则横切阶段 ② 与 ③ 的真机落地与安全收尾。

---

## 关键技术问题

### 1. 骨架拓扑匹配
人类有 23 个主要关节，大多数人形机器人有 40-70 个（含手指）或更少（G1 = 43 DoF，无手指）。

匹配策略：
- **子树对齐**：人类骨架子树 ↔ 机器人骨架子树
- **忽略细节关节**：将人类手指聚合为"手端点"

### 2. 比例缩放（Scale）
人类（1.7m）vs 机器人（0.8m ~ 1.9m）体型不同：
- 末端轨迹按比例缩放：`p_robot = (L_robot / L_human) * p_human`
- 注意脚到地面的高度归一化

### 3. 接触保真度（Contact Consistency）
- 重定向时需要保留"哪只脚在地面"的接触相位
- 否则物理仿真下机器人会穿地或飞起

### 4. 关节限制满足
人类关节活动度（ROM）与机器人关节限制可能不同，需 clip + 后处理优化

---

## 工具与数据集

| 工具/数据集 | 用途 |
|------------|------|
| CMU MoCap Database | 大量人类动作捕捉数据 |
| [AMASS](../entities/amass.md) | 多源光学动捕统一到 **SMPL** 序列的大规模档案（注册下载；许可见官方） |
| [LaFAN1](../entities/lafan1-dataset.md) | Ubisoft 发布的多主题 **BVH** 棚拍动捕（**CC BY-NC-ND**；常用作基准与部署案例数据源） |
| [OMOMO](../entities/omomo-dataset.md) | 人–物交互 MoCap（~10 h / 15 物体）；loco-manipulation 重定向常见上游 |
| [PHUMA](../entities/dataset-bfm-phuma.md) | **已重定向** 的 G1/H1-2 locomotion（~73 h）；PhySINK 物理可信管线 |
| [Humanoid Everyday](../entities/humanoid-everyday-dataset.md) | **真机人形** 多模态操作集（260 任务）；非 MoCap 参考库 |
| [人形数据五集选型](../comparisons/humanoid-reference-motion-datasets.md) | AMASS / LaFAN1 / OMOMO / PHUMA / Humanoid Everyday 对照 |
| [Mixamo](../entities/mixamo.md) | Adobe **商业**角色与动画库；快速原型与美术管线友好，**不等同**于可自由再分发的原始 MoCap 研究档案 |
| [MetaHuman](../entities/metahuman.md) | Epic **UE 数字人**平台（Creator + Animator）；5.8 起单相机全身无标记表演与 **OpenRigLogic（MIT）**；多作视觉/表演源，接入机器人需重定向 |
| SMPL / SMPL-X | 人类体型参数化模型，便于重定向 |
| [SOMA-X](../entities/soma-x.md) | NVIDIA 统一参数化人体拓扑（SMPL/MHR/Anny 等枢纽）；AMASS/SMPL→SOMA 转换 |
| [PHC](../entities/phc.md) | SMPL→人形 fitting 重定向 + 物理模仿控制（AMASS 管线） |
| [SOMA Retargeter](../entities/soma-retargeter.md) | NVIDIA SOMA BVH→G1 CSV，GPU IK（SEED 数据生态） |
| [mocap_retarget](../entities/mocap-retarget.md) | 工程向动捕→机器人脚本参考 |
| [GVHMR](../entities/gvhmr.md) | 单目视频→SMPL 全局人体运动（重定向上游） |
| [VideoMimic](../entities/videomimic.md) | 视频→人形参考 + RL 模仿 |
| [human2humanoid](../entities/human2humanoid.md) | LECAR 遥操栈；含 AMASS 重定向脚本 |
| [motion_imitation（四足）](../entities/motion-imitation-quadruped.md) | 动物 MoCap→四足模仿奠基仓库 |
| [STMR 四足重定向](../entities/stmr-quadruped-retargeting.md) | 空间+时间重定向 + legged_gym RL |
| [Go2 Motion Imitation](../entities/go2-motion-imitation.md) | Go2 专用 retarget_motion + Genesis 训练 |
| [MotionCode](../entities/motioncode.md) | 产业侧人体运动采集与标注（官网宣称可进 Isaac / MuJoCo 等；多作原始运动源，仍需重定向） |
| [FreeMoCap](../entities/freemocap.md) | 开源多相机 USB 动捕与 GUI 平台，适合作为低成本原始人体轨迹来源（AGPL，集成前需合规评估） |
| [fairmotion](../entities/fairmotion.md) | Meta 通用动捕数据处理库（BVH/AMASS IO、3D 变换、FK，已归档）；重定向上游数据层，本身不做机器人重定向 |

---

## 与 AMP / ASE 的关系

AMP 和 ASE 的核心是：从 MoCap 数据中提取运动风格先验，引导 RL 策略。

流程：
```
MoCap 数据 → Motion Retargeting → 机器人参考轨迹
→ 判别器训练（是否像参考） → RL 策略风格约束
```

Motion Retargeting 的质量直接决定 AMP 能学到多自然的动作。

---

## 参考来源
- [KungFuAthleteBot](../../sources/papers/kung_fu_athlete_bot.md)
- [Chasing Autonomy: Dynamic Retargeting and Control Guided RL for Performant and Controllable Humanoid Running](../../sources/papers/chasing_autonomy.md)

- Peng et al., *AMP: Adversarial Motion Priors for Style-Preserving Physics-Based Character Control* (2021) — AMP 中的 motion retargeting 应用
- Choi et al., *SMPL-X: Expressive Whole Body Pose Estimation* (CVPR 2019) — 人体参数化模型
- Liao et al., *Real-Time Motion Retargeting to Highly Varied User-Specific Hand Anatomies* (CHI 2019) — 异构骨架重定向
- **ingest 档案：** [sources/papers/omniretarget_arxiv_2509_26633.md](../../sources/papers/omniretarget_arxiv_2509_26633.md) — OmniRetarget：interaction mesh 交互保留重定向（PHP 等下游的上游）；配套 [holosoma 代码](../../sources/repos/holosoma.md)、[HF 数据集](../../sources/sites/omniretarget-dataset-huggingface.md)
- **ingest 档案：** [sources/papers/teleoperation.md](../../sources/papers/teleoperation.md) — ALOHA / OmniH2O / UMI / AnyTeleop 遥操作系统
- **ingest 档案：** [sources/papers/diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — ACT（CVAE 动作块预测）
- [sources/papers/motion_control_projects.md](../../sources/papers/motion_control_projects.md) — GMR 的总结强调了“运动学重定向之后还需要动力学一致化层”
- **ingest 档案：** [sources/papers/exoactor.md](../../sources/papers/exoactor.md) — ExoActor 的消融提供"视频生成→动作估计→tracking"链路下不引入中间重定向反而更稳的反例
- **ingest 档案：** [sources/papers/neural_motion_retargeting_nmr.md](../../sources/papers/neural_motion_retargeting_nmr.md) — NMR：CEPR 数据管线 + 神经重定向 + G1 全身实验
- **ingest 档案：** [sources/papers/reactor_rl_physics_aware_motion_retargeting.md](../../sources/papers/reactor_rl_physics_aware_motion_retargeting.md) — ReActor：仿真内双层优化 + RL 跟踪的物理感知重定向（arXiv:2605.06593）
- **ingest 档案：** [sources/papers/spider_scalable_physics_informed_dexterous_retargeting.md](../../sources/papers/spider_scalable_physics_informed_dexterous_retargeting.md) — SPIDER：并行仿真采样优化 + 虚拟接触引导的规模化物理重定向（arXiv:2511.09484）
- **ingest 档案：** [sources/papers/toporetarget_arxiv_2606_16272.md](../../sources/papers/toporetarget_arxiv_2606_16272.md) — TopoRetarget：hand–object interaction mesh + Laplacian 灵巧重定向 + 轻量 PPO 跟踪（arXiv:2606.16272）；配套 [项目页](../../sources/sites/toporetarget-github-io.md)
- **ingest 档案：** [sources/papers/bifrost_umi_arxiv_2605_03452.md](../../sources/papers/bifrost_umi_arxiv_2605_03452.md) — BifrostUMI SKR：稀疏关键点 + 仅身高缩放 + mink IK（arXiv:2605.03452）
- **ingest 档案：** [sources/sites/jc-bao-spider-project-github-io.md](../../sources/sites/jc-bao-spider-project-github-io.md) — SPIDER 项目页 jc-bao.github.io/spider-project（管线演示与 BibTeX）
- **ingest 档案：** [sources/sites/amass-dataset.md](../../sources/sites/amass-dataset.md) — AMASS：SMPL 统一人体动捕元数据集（MPI-IS 站点与 ICCV 2019 论文索引）
- **ingest 档案：** [sources/repos/ubisoft-laforge-animation-dataset.md](../../sources/repos/ubisoft-laforge-animation-dataset.md) — LaFAN1：Ubisoft BVH 动捕与评估脚本（SIGGRAPH 2020 论文配套）
- **ingest 档案：** [sources/repos/omomo_release.md](../../sources/repos/omomo_release.md) — OMOMO：人–物交互 MoCap（SIGGRAPH Asia 2023）
- **ingest 档案：** [sources/repos/phuma.md](../../sources/repos/phuma.md) — PHUMA：G1/H1-2 预重定向 locomotion（arXiv:2510.26236）
- **ingest 档案：** [sources/sites/humanoideveryday.md](../../sources/sites/humanoideveryday.md) — Humanoid Everyday：真机开放世界操作集
- **ingest 档案：** [sources/sites/mixamo.md](../../sources/sites/mixamo.md) — Mixamo：Adobe 在线角色与动画服务（商业许可与管线说明）

---

## 关联页面
- [Motion Retargeting Pipeline](./motion-retargeting-pipeline.md) — 端到端工程链路视角：源归一 → 骨架对齐 → IK → 物理筛选 → 配对监督
- [Motion Retargeting Objective（重定向目标函数形式化）](../formalizations/motion-retargeting-objective.md) — 姿态相似 / 末端接触 / 平衡 / 限位 / 平滑项的统一加权和及其三种工程退化
- [Motion Data Quality（动作数据质量维度）](./motion-data-quality.md) — 形态差距/接触/物理/规模四轴决定重定向是否可省略及需补几层
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
- [OmniRetarget](../entities/paper-hrl-stack-03-omniretarget.md) — interaction mesh 硬约束 + 单演示增广的交互保留重定向（ICRA 2026）
- [holosoma](../entities/holosoma.md) — OmniRetarget 开源实现与 WBT/locomotion 训练部署框架
- [OmniRetarget 数据集](../entities/omniretarget-dataset.md) — G1 交互重定向轨迹 HuggingFace 公开子集
- [Mixamo](../entities/mixamo.md) — 商业动画库，与科研向 MoCap 档案对照阅读
- [SAM 3D Body](../entities/sam-3d-body.md) — 单目 RGB 全身 MHR 估计，可作视频→重定向上游
- [SAM3DBody-cpp](../entities/sam3dbody-cpp.md) — 上述模型的 C++/BVH 工程导出
- [BifrostUMI（论文实体）](../entities/paper-bifrost-umi.md) — Robot-Free 示范的 SKR 与全身 WBC 接口
- [PHC](../entities/phc.md) — SMPL fitting 重定向与大规模物理模仿
- [SOMA Retargeter](../entities/soma-retargeter.md) — SOMA/SEED→G1 批处理重定向
- [GVHMR](../entities/gvhmr.md) / [VideoMimic](../entities/videomimic.md) — 视频→人体/人形两条管线入口
- [human2humanoid](../entities/human2humanoid.md) — 遥操与 AMASS 重定向同仓
- [mocap_retarget](../entities/mocap-retarget.md) — 轻量工程向 MoCap 映射参考
- [motion_imitation（四足）](../entities/motion-imitation-quadruped.md) / [STMR](../entities/stmr-quadruped-retargeting.md) / [AMP_for_hardware](../entities/amp-for-hardware.md) / [MetalHead](../entities/metalhead.md) / [LeggedGym-Ex](../entities/leggedgym-ex.md) — 四足模仿与 AMP 生态
- [Go2 Motion Imitation](../entities/go2-motion-imitation.md) — Go2 + Genesis 重定向样板
- [PAN Motion Retargeting](../entities/pan-motion-retargeting.md) / [WalkTheDog](../entities/walk-the-dog.md) — 跨形态（人↔四足）学习式对齐研究
- [AMP-RSL-RL](../entities/amp-rsl-rl.md) — rsl_rl(PPO)+AMP 的人形模仿实现，重定向/动捕产物的消费侧（可 pip 安装）
- [fairmotion](../entities/fairmotion.md) — 上游通用动捕数据处理库（已归档），重定向前的数据 IO/表示层
