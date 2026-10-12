# Draft: A Parametric Tool for Robot Design Exploration

> 来源归档（论文摘录；非逐字全文）

- **标题：** Draft: A Parametric Tool for Robot Design Exploration
- **类型：** paper（机器人参数化建模 / 设计探索 / 强化学习评估）
- **状态：** arXiv preprint v1
- **原始链接：**
  - arXiv abstract: <https://arxiv.org/abs/2609.38405>
  - arXiv HTML: <https://arxiv.org/html/2609.38405v1>
  - Code: <https://github.com/davidhnguyen2000/draft>
  - Supplementary video: <https://youtu.be/egzlEqfwwLU>
- **作者：** David Nguyen, Marcelo Coelho, Sangbae Kim
- **机构：** Massachusetts Institute of Technology (MIT)
- **入库日期：** 2026-10-12
- **一句话说明：** 以显式参数文件和 kinematic tree 快速编译中等保真 MuJoCo MJCF，使用执行器/机器人数据拟合约束设计属性，并通过 twins 与统一 RL 任务评估设计取舍；不包含自动 co-design optimizer。
- **沉淀到 wiki：** [Draft 项目实体](../../wiki/entities/paper-draft-parametric-robot-design.md)

## 论文核心摘录

### 1. 动机与范围

传统 CAD 到仿真描述的交接需要反复导出和训练，使腿长、执行器放置等影响控制上限的决定过早冻结。Draft 将其定位为通用的“中间层”：设计者改规格，生成器快速编译可训练模型，再依据反馈继续设计。

关键范围边界是作者明确排除 outer-loop design optimizer。Draft 旨在给工程师生成模型并提供评估，而不是给出自动寻找最优机器人的端到端共设计算法。

### 2. 规格与编译器

一个机器人由参数文件（长度、密度、电机类别等）、含参数表达式的 kinematic tree、以及根部网格描述组成。统一生成器将维度和执行器选择传播到 visual/collision/inertial/actuation 属性，输出 MuJoCo MJCF。主要构件是基本串联连杆、球形末端、支撑脚板和根部网格。

论文报告 29-joint humanoid 的验证 MJCF 在 laptop CPU 上约 202 ms 编译，12-joint quadruped 约 50 ms。原因之一是树结构表达式使许多尺寸可由其他设计参数推导。

### 3. 执行器与结构趋势

- 执行器目录共 114 个 integrated actuator：QDD 33 个、MidGear 33 个、High-GR 48 个；按传动类型分组，覆盖 2.5–1180 Nm 与 0.19–9.3 kg。
- 拟合关系以峰值扭矩、空载转速和长径比为输入，估计减速比、质量、体积、尺寸和转子惯量。留出误差大致为：减速比 1.33×、质量 1.28×、体积 1.30×；转子惯量因外壳半径与 rotor 类型不明较松，为 1.56×。
- 机器人描述部分从 118 个候选描述中筛出 49 个用于拟合（29 humanoids、20 quadrupeds）；连杆按类型分别拟合质量与长度关系，躯干使用体积密度。
- 四平台 twin 验证的总质量几何平均 fold error 为 1.10×。该结果是模型估计一致性，不是实物制造验证。

### 4. 设计过滤与评估

流程根据预先声明的可行条件筛除方案，例如几何/执行器约束和结构假设。论文构造 Cheetah、Bear、Giraffe 三种四足方案：轻型低减速 QDD、重型中减速高扭矩、以及长腿高减速执行器设计。三者均通过设计过滤器后进入同一两阶段学习课程。

每种设计使用 mjlab 和 PPO，各跑十个随机种子。第一阶段训练速度跟踪基础策略，含轻度不平地形和间歇推扰；第二阶段分别提升峰值速度、地形通过和抗推稳定性。奖励按 Froude similarity 随机体尺度调整，避免固定步幅/速度偏向某一尺寸。

| 设计 | 质量 | 腿长 | 最高速度 | 台阶高度 | 抗推速度 | 特征 |
|---|---:|---:|---:|---:|---:|---|
| Cheetah | 33 kg | 0.48 m | 3.63 m/s | 0.12 m | 7.9 m/s | QDD 6:1，轻快 |
| Bear | 110 kg | 0.74 m | 4.33 m/s | 0.12 m | 8.6 m/s | MidGear 20:1，绝对速度/抗推最高 |
| Giraffe | 91 kg | 1.30 m | 2.17 m/s | 0.20 m | 5.3 m/s | High-GR 80:1，越障最高 |

三种设计没有一个在所有指标上占优。归一化腿长后排序会变化：Cheetah 在相对步高及抗推上有优势；这说明评价口径和设计尺度会影响结论。作者还检查惯量参数扰动后的策略鲁棒性，并分析执行器限幅与任务能力的关系。

### 5. 解释与局限

1. **模型保真度是“中等”。** 基础几何和经验趋势足以做早期比较，但不能代替 CAD/FEA、材料选择、热设计、供应链或安全认证。
2. **经验趋势有覆盖边界。** 执行器样本排除非集成 rotor-stator；链接质量受材料、描述质量及 actuator-link 归属假设影响。
3. **控制评测不是设计优化。** RL 的角色是比较生成方案在任务上的可控性与能力；作者未实现可自动搜索设计变量的外层优化器。
4. **结果不能简单外推。** 详细演示是三个四足方案、特定 mjlab/PPO 课程与任务指标，不代表所有机器人/任务的通用最优排序。
5. **开源许可要分清。** Draft 代码仓为 MIT；README 声明外部厂商 robot models 不随仓直接包含，下载资产另有许可证。

## 对知识库的映射

- [Draft 实体](../../wiki/entities/paper-draft-parametric-robot-design.md)：工程动机、生成链、拟合数据、实验结果、局限和复现入口。
- [MuJoCo 实体](../../wiki/entities/mujoco.md)：输出模型格式与仿真基础设施。
- [Shape Your Body 实体](../../wiki/entities/paper-shape-your-body-value-gradient-design.md)：与显式优化器路线对照。

## 核对状态

- [x] 论文作者、摘要和代码链接已核对
- [x] 方法、数据范围、验证和 RL 实验来自 arXiv HTML v1
- [x] 明确区分生成/评估工具与自动优化算法
