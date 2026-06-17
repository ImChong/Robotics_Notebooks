# MotionDisco（arXiv:2606.06139）

> 来源归档（ingest）

- **标题：** MotionDisco: Motion Discovery for Extreme Humanoid Loco-Manipulation
- **缩写：** **MotionDisco**
- **类型：** paper / motion-discovery / humanoid loco-manipulation / task-and-motion-planning
- **arXiv：** <https://arxiv.org/abs/2606.06139>（HTML：<https://arxiv.org/html/2606.06139v1>）
- **PDF：** <https://arxiv.org/pdf/2606.06139>
- **项目页：** <https://atarilab.github.io/motiondisco.io/>
- **视频：** <https://youtu.be/DHiVz34QYlw>
- **作者：** Ilyass Taouil†, Michal Ciebelski†, Shafeef Omar†, Haizhou Zhao, Angela Dai, Aaron M. Johnson, Majid Khadiv（† 共同一作）
- **机构：** Technical University of Munich（TUM）；New York University（NYU）；Carnegie Mellon University（CMU）
- **入库日期：** 2026-06-17
- **一句话说明：** 不依赖遥操作或人体动作重定向，用 **LLM 引导的进化式程序搜索** 在离散 **接触模式序列** 空间中发现长时程、接触丰富的人形 loco-manipulation 轨迹；**分层运动学剪枝 + 接触显式 kinodynamic 轨迹优化** 返回数值与文本反馈闭环指导变异；发现轨迹经 **DeepMimic 式 RL 跟踪** 在 **Unitree G1** 真机零样本部署——据作者称首个完全通过自动化进化搜索发现并执行此类长时程 loco-manipulation 真机行为的工作。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §1）

- **链接：** <https://arxiv.org/abs/2606.06139>；<https://atarilab.github.io/motiondisco.io/>
- **核心贡献：** 长时程人形 loco-manipulation 的接触交互空间随任务步长与场景物体数 **组合爆炸**；主流路线依赖 **人体演示重定向** 或 **遥操作** 绕过探索难题，但难以覆盖任意新场景且把机器人限制在类人动作。MotionDisco 提出 **从零发现（motion discovery）**：LLM 迭代变异生成 **接触计划 Python 程序**，由 **顺序运动学可行性检查** 快速剔除几何不可行序列，再通过 **全动力学轨迹优化** 评估动态可行性；优化器把失败点翻译为 **文本反馈** 注入下一轮 LLM 变异，形成 **推理–优化闭环**。真机上对发现轨迹训 **RL 跟踪策略**（DeepMimic 奖励 + 域随机化）并 **零样本执行**。
- **对 wiki 的映射：**
  - [MotionDisco 论文实体](../../wiki/entities/paper-motiondisco-extreme-humanoid-loco-manipulation.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
  - [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)（对照：无演示上游）

### 2) 接触显式运动规划模块（§3.1）

- **链接：** <https://arxiv.org/html/2606.06139v1> §3.1
- **核心贡献：** 将每个时刻的 **接触模式** $c$ 定义为各接口（末端、足底、可动物体）的 **单边接触分配**；给定固定接触模式序列 $\mathcal{C}$，在连续时间上求解状态/控制/接触力矩与阶段时长，约束动力学、接触、碰撞与关节限位。评估分两档：**(1)** 顺序运动学优化（忽略动力学，检验碰撞、限位与切换一致性）作 **快速几何过滤器**；**(2)** 通过运动学检验后运行 **全 kinodynamic TO**（direct multiple shooting；**acados** / **Hippo** 求解器）产出可用于策略训练的参考轨迹。不可行时定位 **最长可行前缀** 与首个失败模式/切换，生成结构化文本反馈。
- **对 wiki 的映射：**
  - [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)

### 3) LLM 引导进化式程序搜索（§3.2）

- **链接：** <https://arxiv.org/html/2606.06139v1> §3.2；Appendix A
- **核心贡献：** 接触计划表示为可执行 **Python 程序** $p_n$，输出离散接触模式序列；在 **LLM 引导进化树** 上迭代：选父节点 → LLM 变异程序（条件于场景文本 $\mathcal{S}$、目标 $\mathcal{G}$、父节点分数与反馈 $\tau_n$）→ 运动规划评分 → 回插子节点。树扩展采用 **ShinkaEvolve**（ICLR 2026）的 **岛屿分层种群 + 新颖性拒绝采样**，避免早熟收敛。LLM 组件使用 **Claude Opus 4.7**。程序 API 提供 `get_initial_mode()`、`walk(n_steps)`（直线平地步态，不自动绕障）、`append_mode()`（抓取/放置/登高等 **有趣子目标**），把普通行走委托给运动学/TO 层。
- **对 wiki 的映射：**
  - [Task and Motion Planning（Paper Notebooks 占位）](../../wiki/entities/paper-notebook-task-and-motion-planning-for-humanoid-loco-manip.md)
  - [Teleoperation](../../wiki/tasks/teleoperation.md)（对照：无人类在环采集）

### 4) 八项任务评测与消融（§4.1–4.2）

- **任务（8）：** Banana、Box Stacking、Climb Table w/ Box、Long-Dist. Pick & Place、Move Through Clutter、Parkour Pick & Place 1/2、Under-Table Pick & Place——从简单 pick-place 到 **攀台、穿障、桌下操作** 等全身 loco-manip。
- **消融（表 1）：** **单次 LLM 调用（SC）** 在多项任务失败；**MotionDisco 无文本反馈（MD w/o TF）** 可解全场景但 valid% 与 TO cost 较差；**完整 MotionDisco（MD）** 在相同搜索预算下 **valid% 更高、TO cost 更低**；**首条可行解** 在 **1.3–7.5 分钟** 量级出现。进化搜索还在 **单次运行** 内为同一任务产出 **多样接触计划**（如 Parkour Pick & Place 2 的不同手足轨迹）。
- **对 wiki 的映射：**
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

### 5) 真机部署与局限（§4.3 / §6）

- **真机：** **Unitree G1**；对发现轨迹训 **DeepMimic 式 RL 跟踪** + 域随机化，**零样本** 真机复现；作者报告各尝试任务 **多次连续成功**。
- **局限：** 当前仅支持 **矩形贴片单边粘附接触**；物体限于 **刚体箱状**；假设 **已知场景文本描述**、无感知模块；未来方向包括滑动/非单边接触、关节物体、视觉建场景与仿真就绪场景构建（SimRecon / HoloScene 等）。
- **对 wiki 的映射：**
  - [Unitree G1](../../wiki/entities/unitree-g1.md)
  - [Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md)
  - [DynaRetarget（同团队相关重定向线）](../../wiki/entities/paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md)

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-motiondisco-extreme-humanoid-loco-manipulation.md`](../../wiki/entities/paper-motiondisco-extreme-humanoid-loco-manipulation.md)
- 互链参考：[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)、[Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)
