# SkillMimic: Learning Basketball Interaction Skills from Demonstrations（arXiv:2408.15270）

> 来源归档（ingest）

- **标题：** SkillMimic: Learning Basketball Interaction Skills from Demonstrations
- **类型：** paper / physics-based-animation / human-object-interaction / imitation-learning / hierarchical-rl / basketball
- **arXiv abs：** <https://arxiv.org/abs/2408.15270>
- **arXiv HTML：** <https://ar5iv.labs.arxiv.org/html/2408.15270>
- **PDF：** <https://arxiv.org/pdf/2408.15270>
- **Venue：** CVPR 2025 Highlight
- **项目页：** <https://ingrid789.github.io/SkillMimic/> — 归档见 [`sources/sites/ingrid789-skillmimic-github-io.md`](../sites/ingrid789-skillmimic-github-io.md)
- **代码：** <https://github.com/wyhuai/SkillMimic> — 归档见 [`sources/repos/skillmimic.md`](../repos/skillmimic.md)
- **机构：** Hong Kong University of Science and Technology（HKUST）、Unitree Robotics、Peking University、Tsinghua University、International Digital Economy Academy（IDEA）、Tencent、Carnegie Mellon University（CMU）
- **分类（Paper Notebooks）：** 13_Physics-Based_Animation
- **入库日期：** 2026-07-28
- **一句话说明：** 用 **统一 HOI 模仿奖励 + Contact Graph** 从人–球演示学可复用篮球交互技能（运球/上篮/投篮等），再以 **高层控制器（HLC）** 组合技能完成连续得分等长程任务；Isaac Gym 开源训练/评测与 BallPlay-M 子集。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://ingrid789.github.io/SkillMimic/> | 演示视频、技能切换、HLC 任务、3D 可视化 |
| 代码仓库 | <https://github.com/wyhuai/SkillMimic> | 训练/评测、预训练权重、BallPlay-M 子集、Blender 渲染 |
| 同作者 HOI 线 | [PhysHOI](https://github.com/wyhuai/PhysHOI) | 动态人–物交互物理模仿前作 |
| 后续 | [SkillMimic-V2](https://github.com/Ingrid789/SkillMimic-V2) | 稀疏/噪声演示下的鲁棒交互技能 |
| 代码基座 | [ASE](https://github.com/nv-tlabs/ASE) | NVIDIA ASE；SkillMimic 实现基于 ASE + PhysHOI |
| 篮球组合对照 | [Learning to Ball](https://arxiv.org/abs/2509.22442) | 多子技能策略 + 软路由器拼长程篮球连招 |
| Paper Notebooks 进度锚点 | [`humanoid_pnb_skillmimic-learning-basketball-interaction-skill.md`](./humanoid_pnb_skillmimic-learning-basketball-interaction-skill.md) | 姊妹仓库 progress 条目溯源 |

## 摘要级要点

- **问题：** 传统 HOI / 交互技能 RL 依赖 **逐技能手工奖励**，难跨技能泛化；篮球运球、上篮、投篮等需要 **人–球接触与实时调整**。
- **方法：** **SkillMimic** — 把技能定义为 **参考 HOI 状态转移集合**，用 **统一配置** 的模仿奖励学 **单一 skill policy**；技能多样性与泛化随数据集增大而提升；测试时可不依赖参考即可连续运行，并支持 **数据集中未出现的技能切换**。
- **Contact Graph（CG）+ CGR：** 显式建模交互接触；相对仅模仿身体/物体运动，CGR 对精确接触模仿关键；消融强调 **CG 奖励** 与奖励项 **乘法组合**。
- **分层：** 预训练 skill policy（LLC）后，训 **High-Level Controller（HLC）** 输出 skill label \(c_t\)，用极简任务奖励完成 heading / circling / scoring / throwing。
- **数据：** **BallPlay-V**（单目 RGB 估计，8 类基础技能，检验噪声鲁棒）+ **BallPlay-M**（光学 MoCap，约 35 分钟多样技能）；仓库已释出 BallPlay-M **子集**，完整原始数据与处理代码仍 TODO。
- **仿真：** Isaac Gym；SMPL-X 类人形；RSI；固定最大 episode 长度以平衡不同 clip 奖励上界。

## 核心摘录（面向 wiki 编译）

### 1) 技能 = HOI 状态转移模仿

- 若人形操控物体使 **HOI 状态转移** 贴近参考，则视为学会该技能。
- 相对逐技能手工 reward（DeepMimic / AMP / ASE 族 + 交互专项），SkillMimic 强调 **data-driven、skill-agnostic、可扩展**。

### 2) 统一 HOI 模仿奖励（结构级）

| 分量 | 作用 |
|------|------|
| \(r_t^{b}\) | 身体运动模仿 |
| \(r_t^{o}\) | 物体（球）运动模仿 |
| \(r_t^{rel}\) | 人–物相对位置 |
| \(r_t^{cg}\) | Contact Graph 接触模仿（CGR） |

- 设计目标：**不引入技能专用超参**；消融显示去掉 CGR 或改乘法为加法会显著伤害 Acc. / MPJPE / \(E_{cg}\)。

### 3) 观测与条件

- 身体：root 高度 + root 局部系位姿/速度/角速度（proprioception）。
- 手指等 **净接触力**；物体状态；**skill label** \(c\) 条件化多技能。
- HLC：额外任务观测 \(h_t\)（如篮筐位置）→ 预测 \(c_t\) 驱动冻结/预训练 skill policy。

### 4) HLC 任务（项目页 / 仓库）

| 任务 | 含义 |
|------|------|
| Heading | 运球至目标点 |
| Circling | 绕中心以目标半径运球 |
| Scoring | 运球–上篮得分–抢篮板–重复 |
| Throwing | 抛球达目标高度 |

### 5) 开源核查（项目页 + GitHub，截至 2026-07-28）

| 组件 | 状态 |
|------|------|
| 训练与评测代码（`skillmimic/run.py`） | ✅ 已开源（Apache-2.0） |
| 预训练 LLC / HLC checkpoints | ✅ `skillmimic/data/models/` |
| BallPlay-M 子集 | ✅ `skillmimic/data/motions/BallPlay-M/` |
| Blender 渲染 | ✅ `blender_for_SkillMimic/` |
| 完整 BallPlay-M 原始数据与处理代码 | ⬜ README TODO |

## 对 wiki 的映射

- 主实体页：[paper-notebook-skillmimic-learning-basketball-interaction-skill](../../wiki/entities/paper-notebook-skillmimic-learning-basketball-interaction-skill.md)
- 分类父节点：[paper-notebook-category-13-physics-based-animation](../../wiki/overview/paper-notebook-category-13-physics-based-animation.md)
- 方法背景：[ASE](../../wiki/methods/ase.md)、[Hierarchical RL](../../wiki/methods/hierarchical-reinforcement-learning.md)、[Imitation Learning](../../wiki/methods/imitation-learning.md)
- HOI / 篮球对照：[InterMimic](../../wiki/entities/paper-bfm-15-intermimic.md)、[Learning to Ball](../../wiki/entities/paper-notebook-learning-to-ball.md)

## 参考来源（原始）

- Wang et al., *SkillMimic: Learning Basketball Interaction Skills from Demonstrations*, CVPR 2025 Highlight. <https://arxiv.org/abs/2408.15270>
- 项目页：<https://ingrid789.github.io/SkillMimic/>
- 代码：<https://github.com/wyhuai/SkillMimic>
