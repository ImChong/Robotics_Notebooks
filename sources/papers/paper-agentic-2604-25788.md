# KinDER: A Physical Reasoning Benchmark for Robot Learning and Planning

- **作者：** Yixuan Huang\*、Bowen Li\*、Vaibhav Saxena\*、Yichao Liang、Utkarsh Aashu Mishra、Liang Ji、Lihan Zha、Jimmy Wu、Nishanth Kumar、Sebastian Scherer、Danfei Xu、Tom Silver（\* 共同一作）
- **机构：** Princeton University、Carnegie Mellon University、Georgia Tech、University of Cambridge、NVIDIA、MIT
- **版本 / 载体：** arXiv:2604.25788v2 [cs.RO]（2026-05-04，v1 提交于 2026-04-28）；已被 Robotics: Science and Systems (RSS) 2026 接收（21 页，8 图）
- **原文：** [arXiv HTML v2](https://arxiv.org/html/2604.25788) / [arXiv 摘要页](https://arxiv.org/abs/2604.25788) / [项目页](https://prpl-group.com/kinder-site/)
- **项目页 / 代码：** 已开源 — 项目页 [prpl-group.com/kinder-site](https://prpl-group.com/kinder-site/)；环境核心仓库 [kindergarden](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden)（MIT 协议，PyPI 包名 `kindergarden`）；基线实现仓库 [kinder-baselines](https://github.com/Princeton-Robot-Planning-and-Learning/kinder-baselines)；预训练模型与演示数据集发布于 [Hugging Face kinder-bench](https://huggingface.co/kinder-bench) 与 [kinder-datasets](https://huggingface.co/datasets/kinder-bench/kinder-datasets)
- **核对范围：** 已通读 arXiv v2 HTML 全文（§I–§VIII 全部正文、Table I–V、附录 A–E 含 Table VI–XII、Fig. 7–8 提示模板）及项目主页全文（208 行），并对照 `kindergarden`、`kinder-baselines` 与 `prpl-mono` 仓库的 README 与目录采样
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[KinDER 实体页](../../wiki/entities/paper-agentic-2604-25788.md)

### 1. 研究动机与五类核心物理推理挑战（§I–§III, Table I）

- **去耦合评测目标：** 现有机器人基准（如 BEHAVIOR-1k、CALVIN、LIBERO、RoboCasa、ManiSkill-HAB 等）侧重家庭应用或语言/视觉泛化，物理推理与感知、语言理解及任务复杂度高度耦合；而 PHYRE、I-PHYRE、Virtual Tools 等纯物理推理基准面向 2D 谜题或人类认知，不具备真实机器人形态与操控约束。KinDER 旨在将「运动学与动力学具身推理（Kinematic and Dynamic Embodied Reasoning）」从复杂感知与语义中解耦出来。
- **五类核心挑战：**
  1. **基础空间关系（Basic Spatial Relations）：** 被动判别与主动布置物体相对空间位置（如左右前后）。
  2. **非抓持多物体操作（Nonprehensile Multi-Object Manipulation）：** 推、拉、扫、舀、倒以及利用全身/整臂接触同时操作多个物体。
  3. **工具使用（Tool Use）：** 利用杆、钩、箱、扫帚、铲子等工具间接作用于目标物体。
  4. **组合几何约束（Combinatorial Geometric Constraints）：** 物体数量增长时，避碰与空间可放置性约束呈组合级增长（如狭窄货架、杂乱遮挡清理）。
  5. **动力学约束（Dynamic Constraints）：** 受速度、惯性、倾倒、抛掷或杠杆平衡等动力学限制的任务（如不倾洒舀倒、越障抛投、平衡木配重）。

### 2. KinDERGarden 25 个环境架构（§IV, Appendix A Table IX–XII）

- **统一接口：** 继承 Gymnasium API（`reset()` 程序化生成无限任务变体，`step()` 返回稀疏奖励：每步 `-1` 直到目标达成终止）。
- **以对象为中心的状态表示（Object-Centric States）：** 将对象名（如 `robot`、`hook`）映射为按类型定义的实值特征向量；例如 `MobileManipulator` 包含底座 $\mathrm{SE}(2)$ 位姿与速度、$\mathbb{R}^7$ 机械臂关节状态与速度及 $[0,1]$ 夹爪状态；`Movable` 包含 $\mathrm{SE}(3)$ 位姿/速度及 $\mathbb{R}^3$ 包围盒尺寸。同步提供 RGB 图像观测与固定物体数变体的扁平向量接口，并配套 `NoisyObservation` / `NoisyAction` 包装器。
- **四大类别：**
  - **Kinematic2D（6 个）：** 纯 Python 实现，无物理引擎；圆形底座 $\mathrm{SE}(2)$ + 1D 伸缩臂 + 矩形吸盘，发生碰撞时回退状态。包含 `Motion2D`、`Obstruction2D`、`ClutteredRetrieval2D`、`ClutteredStorage2D`、`PushPullHook2D`、`StickButton2D`（动作空间 $\mathcal{A}\in\mathbb{R}^5$）。
  - **Dynamic2D（4 个）：** Pymunk 2D 物理引擎，建模拟合速度/加速度/摩擦/重力，两指夹爪。包含 `DynPushT`（仓库称 `DynPushT2D`）、`DynObstruction2D`、`DynPushPullHook2D`、`ScoopPour2D`（仓库称 `DynScoopPour2D`），动作空间 $\mathcal{A}\in\mathbb{R}^5$。
  - **Kinematic3D（5 个）：** PyBullet 运动学与碰撞检测，无动力模拟；统一采用 TidyBot++ 全向底座 + 7-DOF Kinova Gen3 臂 + Robotiq 2F-85 夹爪（闭合后刚性附着，碰撞回退）。包含 `BaseMotion3D`、`Obstruction3D`、`Packing3D`、`Table3D`、`Transport3D`（动作空间 $\mathcal{A}\in\mathbb{R}^{11}$）。
  - **Dynamic3D（10 个）：** MuJoCo 动力学仿真，动态摩擦抓持（非刚性绑定），配置驱动共享代码，借鉴 BDDL 规范并复用 RoboCasa 物体资产与 MimicLabs 场景资产。Table XII 详列 9 个环境：`Rearrange3D`、`Tossing3D`、`BalanceBeam3D`、`ConstrainedCupboard3D`、`Shelf3D`、`SweepSimple3D`、`SweepIntoDrawer3D`、`SortClutteredBlocks3D`、`ScoopPour3D`；项目页与仓库另包含第 10 个环境 `Dynamo3D`（动作空间 $\mathcal{A}\in\mathbb{R}^{11}$）。

### 3. KinDERGym 软件基础设施与技能抽象（§V, Appendix B）

- 严格工程规范：近 400 个单元测试、CI、lint 与类型检查，测试覆盖 Python 3.10–3.12 及 Ubuntu / macOS / Windows 10。
- **参数化技能与概念：** 技能实现为带 PDDL 算子与连续参数采样器（sampler）的 options（例如 $\mathrm{Pick(object, \theta)}, \theta\in\mathrm{SE}(3)$）；概念实现为接地于对象中心状态的关系谓词分类器（例如 $\mathrm{On(object, surface)}$），两者共同构成两层场景图供分层规划与大模型规划使用。
- **遥操作与演示数据：** 2D 支持键鼠与 PS5 手柄；3D 支持基于 TidyBot++ 陀螺仪/加速度计的 iPhone Web App 及 Meta Quest 3S VR 手柄。论文结合遥操作与规划在 10 个环境中提供每个 $\ge 100$ 条演示轨迹。

### 4. KinDERBench 评测协议与核心数据（§VI, Table II–VIII, Appendix C）

- **8 个代表性测试环境：** `Motion2D-p0`、`StickButton2D-b1`、`DynObstruction2D-o1`、`DynPushPullHook2D-o5`、`BaseMotion3D-o0`、`Transport3D-o2`、`Shelf3D-o1`、`SweepIntoDrawer3D-o5`。每组评估 5 个随机种子 × 每种子 50 回合。
- **13 类基线与全 8 环境平均成功率（SR）：**
  - 分层双层规划 BP（0.57）> 少样本上下文 LLMCon / VLMCon（均为 0.43）> 零样本 LLMPlan / VLMPlan（均为 0.34）> 真值模型采样预测控制 MPC（0.32）= 微调 VLA $\pi_{0.5}$（0.32）> 扩散技能链 GSC（0.26）> 扩散策略加环境状态 DPES（0.25）> 纯图像+本体状态扩散策略 DP（0.24）> PPO（0.13）> 神经转移模型 MBRL（0.08）> SAC（0.02）。
- **关键实验发现：**
  - LLMPlan 与 VLMPlan、LLMCon 与 VLMCon 成绩几乎一致，表明当前 VLM 未能有效利用附加 RGB 图像增益；上下文示例（in-context examples）明显提升推理规划表现（0.34 → 0.43）。
  - 同样提供环境状态的 DPES（0.25）并未超越纯视觉 DP（0.24），甚至在 `SweepIntoDrawer3D`（0.04 vs 0.14）与 `Shelf3D`（0.09 vs 0.13）更低。
  - 在多障碍工具使用动态任务 `DynPushPullHook2D-o5` 上，微调 $\pi_{0.5}$ VLA 是唯一取得非平凡成功率（0.43）的基线，其余均接近或等于 0。
  - 在需要开抽屉→抓刷子→清扫入屉的 `SweepIntoDrawer3D-o5` 上，所有方法成功率均 $\le 0.14$（Table III 显示 DP 开抽屉 0.87、抓刷子 0.74，但全部扫入仅 0.14）。
  - `DynObstruction2D` 障碍物数量 OOD 泛化（1 个训练，0/1/2/3 测试，Table IV）：VLA 从 0.50 保持到 0.52 / 0.43 / 0.40，DP 从 0.33 变为 0.35 / 0.30 / 0.26。
  - `StickButton2D` 按钮数量扩展（1→3→5→10，Table V）：BP 成功率从 0.99 骤降至 0.26 / 0.02 / 0.00，单回合推理耗时从 1.51s 增至 38.39s。
  - 噪声注入（Table VIII）：极小观测噪声（stdev=0.01）即导致 BP 在 `StickButton2D` 上 SR 从 1.00 跌至 0.22、`Motion2D` 从 1.00 跌至 0.40。

### 5. 真机验证与局限（§VII–§VIII, Appendix D–E）

- **Real-to-Sim-to-Real 实验：** 在实体 TidyBot++ 上利用俯视标定相机定位底座，通过开放词汇目标检测获取 2D 包围盒（固定 z 轴尺寸）构建 `Shelf3D` 仿真孪生状态，运行 BP 生成规划轨迹后回传真机按路点执行。
- **原文承认的局限：** 仿真动力学仍与真实物理接触存在偏差；为聚焦中层物理推理，暂排除了随机性、部分可观测性、跨多机器人本体及多智能体协作；基线选择具有代表性但未覆盖所有新兴方法。
