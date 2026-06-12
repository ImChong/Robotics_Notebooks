# VisualMimic: Visual Humanoid Loco-Manipulation via Motion Tracking and Generation（arXiv:2509.20322）

> 来源归档（ingest）

- **标题：** VisualMimic: Visual Humanoid Loco-Manipulation via Motion Tracking and Generation
- **类型：** paper / humanoid / loco-manipulation / visual-rl / sim2real / hierarchical-control / teacher-student
- **arXiv abs：** <https://arxiv.org/abs/2509.20322>
- **arXiv HTML：** <https://arxiv.org/html/2509.20322>
- **PDF：** <https://arxiv.org/pdf/2509.20322>
- **项目页：** <https://visualmimic.github.io/> — 归档见 [`sources/sites/visualmimic-github-io.md`](../sites/visualmimic-github-io.md)
- **代码：** <https://github.com/visualmimic/VisualMimic> — 归档见 [`sources/repos/visualmimic.md`](../repos/visualmimic.md)
- **机构：** Stanford University（Shaofeng Yin*、Yanjie Ze*、Hong-Xing Yu；C. Karen Liu†、Jiajun Wu† 共同指导；* 同等贡献，† 同等指导）
- **分类（Paper Notebooks）：** 04_Loco-Manipulation_and_WBC
- **入库日期：** 2026-06-12
- **一句话说明：** **视觉 Sim2Real 分层框架**：任务无关 **关键点跟踪低层**（人类动作 teacher–student 蒸馏）+ 任务专用 **视觉关键点生成高层**（特权状态 teacher → 深度 visuomotor student），在真机 **零样本** 完成多样 loco-manipulation 并泛化到户外。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://visualmimic.github.io/> | 真机/仿真视频、消融、接口设计对比 |
| 代码仓库 | <https://github.com/visualmimic/VisualMimic> | Sim2Sim 管线与真机任务 checkpoint（部分开源） |
| 运动跟踪同族 | [TWIST](https://arxiv.org/abs/2505.02833) | 低层 motion tracking 奖励结构来源；Yanjie Ze 同作者线 |
| 重定向 | GMR [ze2025gmr] | AMASS、OMOMO → 人形参考 |
| 视觉 sim2real 对照 | [VideoMimic](https://videomimic.github.io/) | 偏环境交互（坐、爬梯），缺全身物体操作 |
| 残差 loco-manip | [ResMimic](https://arxiv.org/abs/2510.05070) | GMT+残差、依赖 MoCap 物体参考；VisualMimic 强调 **无配对人–物数据** |
| 视觉 loco-manip 对照 | [VIRAL](https://arxiv.org/abs/2511.15200) | 规模化 RGB 蒸馏 + WBC API；VisualMimic 用 **关键点接口 + 深度** |
| Paper Notebooks 进度锚点 | [`humanoid_pnb_visualmimic.md`](./humanoid_pnb_visualmimic.md) | 姊妹仓库 progress 条目溯源 |

## 摘要级要点

- **问题：** 非结构化环境中的人形 loco-manipulation 需要 **第一视角感知 + 全身灵巧**；现有路线或依赖 **外置 MoCap**、或 **模仿学习数据稀缺**、或 **视觉 RL 仅能做简单环境交互**（坐、楼梯），难以达到人类级 **全身–物体** 交互。
- **方法：** **VisualMimic** = **低层关键点跟踪器** $\pi_{\mathrm{tracker}}$（任务无关，人类动作先验）+ **高层关键点生成器** $\pi_{\mathrm{generator}}$（任务专用，egocentric 深度 + 本体）。
- **低层（两阶段 teacher–student）：** (1) **Motion tracker 教师**：特权未来 **2 s** 全身参考 + 足端接触力等，PPO + TWIST 同族 $r_{\mathrm{motion}}$；(2) **Keypoint tracker 学生**：仅本体 + 关键点命令 $c^{\mathrm{kp}}_t$（root + 头/双手/双足共 5 点相对误差），**DAgger** 蒸馏。数据：GMR 重定向 **AMASS + OMOMO**。
- **高层（两阶段 teacher–student）：** (1) **状态教师**：特权 **物体状态** + 本体，PPO + 轻量任务奖励（approach、forward progress 等）；(2) **视觉学生**：**深度图 + 本体**，从教师蒸馏；仿真中对深度 **重度 masking** 逼近真机噪声。
- **训练稳定：** (1) 低层训练时 **注入噪声** 适应高层命令；(2) 高层动作 **clip 到人类动作空间（HMS）** 统计，避免 RL 探索超出低层可跟踪范围。
- **真机（论文报告）：** 零样本 sim-to-real；**0.5 kg 箱抬至 1 m**；**3.8 kg 大箱** 全身 steady push；**足球盘带**；**双脚交替踢球**；**户外**（光照、地面变化）仍稳定。
- **对比轴（Table I）：** 相对 TWIST（无视觉策略）、VideoMimic（无 loco-manip/全身 dex）、Hitter（无视觉/全身 dex）等，VisualMimic 同时满足 **Whole-Body Dex + Loco-Manipulation + Visual Policy**。

## 核心摘录（面向 wiki 编译）

### 1) 分层接口：关键点命令

$$c^{\mathrm{kp}}_t=[\Delta p_t,\,\Delta x_t^1,\ldots,\Delta x_t^5]$$

- $\Delta p_t$：root 位置误差（desired − current）。
- $\Delta x_t^i$：头/双手/双足相对 root 的期望−当前误差（$i=1..5$）。
- 设计动机：**紧凑且表达力强**，高层只需生成 6-DOF 级接口，低层承担全身平衡与类人运动。

### 2) 低层 tracker 蒸馏动机

- 直接训 keypoint tracker 可跟踪目标点，但 **动作不够类人**（Fig. 6 消融）。
- 先用 **全参考 motion teacher** 学精确跟踪，再蒸馏到 **仅关键点命令** 的 deployable student。

### 3) 高层任务奖励（示例）

| 组件 | 形式 | 用途 |
|------|------|------|
| **Approach** $R_{\mathrm{approach}}$ | $\exp(-0.1 d)$ 或双手 harmonic mean | 手/足接近物体目标点 |
| **Forward progress** $R_{\mathrm{forward}}$ | $\tanh(10\Delta x_{\mathrm{obj}})$ | 奖励物体向前推进 |

- 任务示例：push / reach / kick / lift / dribble；**无需配对人–物 MoCap**。

### 4) 视觉 Sim2Real

- 学生策略输入：**egocentric depth** + proprioception。
- 仿真训练：**heavy depth masking** 模拟真机深度噪声与视觉 gap（Fig. 8）。
- 部署：**零样本** 真机，无需外置物体状态估计。

### 5) 开源与复现（GitHub，截至 2025-09-24）

| 组件 | 状态 |
|------|------|
| Sim2Sim pipeline | 已发布 |
| 真机任务 checkpoints | 已发布 |
| Low-level tracker 训练代码 | 待发布 |
| High-level 训练代码 | 待发布 |
| Sim2Real 全流程 | 待发布 |

- 环境：Ubuntu 20.04；`conda` Python 3.8；`sim2sim/sim2sim.py --task kick_ball|kick_box|push_box|lift_box`。

## 对 wiki 的映射

- [paper-notebook-visualmimic](../../wiki/entities/paper-notebook-visualmimic.md) — 论文实体页（策展归纳）
- [loco-manipulation](../../wiki/tasks/loco-manipulation.md) — 视觉分层 sim2real 路线
- [paper-twist](../../wiki/entities/paper-twist.md) — 运动跟踪奖励与同作者 TWIST 线
- [videomimic](../../wiki/entities/videomimic.md) — 视觉人形交互对照（偏环境非物体）
- [paper-resmimic](../../wiki/entities/paper-resmimic.md) — GMT/残差 loco-manip 对照
- [paper-viral-humanoid-visual-sim2real](../../wiki/entities/paper-viral-humanoid-visual-sim2real.md) — 规模化 RGB 视觉 sim2real 对照
- [paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md) — Paper Notebooks 分类父节点

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2509.20322>
- 项目页：<https://visualmimic.github.io/>
- 代码：<https://github.com/visualmimic/VisualMimic>
- Paper Notebooks progress：<https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/progress.json>
