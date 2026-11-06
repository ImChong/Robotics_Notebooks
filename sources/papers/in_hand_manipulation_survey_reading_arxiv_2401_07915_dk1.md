# 机器人手内操作的学习方法综述通读（灵巧手运动学视角）

> 来源归档（原文通读 · 支撑 `wiki/concepts/dexterous-kinematics.md`）

- **标题：** Survey of Learning-based Approaches for Robotic In-Hand Manipulation
- **作者：** Abraham Itzhak Weinberg, Alon Shirizly, Osher Azulay, Avishai Sintov
- **类型：** paper / survey / in-hand-manipulation
- **原文：** <https://arxiv.org/abs/2401.07915>（2026-11 通过 arXiv PDF v2 获取全文）
- **一句话说明：** 以"解析建模难以应对复杂环境与不确定性"为出发点，把手内操作按是否保持连续接触分类，并综述模型学习、强化学习、模仿学习三条路线。
- **核对范围：** 通读摘要与 §I–§VII 正文（概述、手内操作类型、非灵巧手与灵巧手、视觉 / 触觉感知、仿真与数据集、模型学习、RL、IL、挑战与结论）；Table I / II 与参考文献列表只浏览，未逐条核对被引文献。本文不含抓取矩阵 / 手指雅可比的推导，经典闭链公式仍依赖 Murray–Li–Sastry 教材（未在此次通读范围内）。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **灵巧 / 非灵巧（§II-A）：** 灵巧操作指多指（或多臂）协同操作物体，灵巧手内操作是靠手自身机构改变物体位姿，通常需要较多自由度、多为拟人手；平行夹爪等低自由度手须借助外部（extrinsic）动作。
- **灵巧手判据（§II-D）：** 引 Mason 与 Salisbury：刚性手至少 3 指、每指 3 关节才可对物体获得可控性。
- **按接触分类（§II-B）：** 保持连续接触的有滚动（rolling）、枕转（pivoting）、滑动（sliding）、手内（in-grasp，利用手的运动学冗余同时改变位置与朝向，可能伴随滑动或滚动接触）与指步态（finger gaiting，在保持力封闭抓取下切换接触位置，通常准静态执行，需要足够多自由度）；不保持连续接触的有 pick-and-place 与动态重抓。强调保持连续接触时必须全程保证抓取稳定。
- **解析模型的困难（摘要、§III）：** 解析建模与按路径编程难以应对复杂环境和不确定性；刚性手的前向运动学已知，转移模型有解析解，柔顺 / 软手则难以获得精确解析模型，需要数据驱动模型。
- **状态表示与转移模型（§III）：** 转移模型 x_{t+1}=f(x_t,a_t)（或概率形式），多靠非线性回归；视觉标记、RGB-D、触觉（含光学触觉）融合用于位姿估计；数据采集覆盖不均、需"好模型才能学好模型"。
- **强化学习（§IV）：** 域随机化（含 OpenAI 的自动域随机化 ADR）是 sim-to-real 主流手段；Isaac Gym 等仿真接触力不可靠，有工作只用二值触觉信号缓解；不可逆状态与回合重置问题；actor-critic、师生蒸馏（特权信息教师）、分层控制（低层模型控制器 + 高层学习策略）、课程学习；触觉可提升学习效率与遮挡下表现。总结：RL 策略在多任务与少样本泛化上仍有困难。
- **模仿学习（§V）：** 行为克隆（BC）与逆强化学习（IRL）；遥操作、手套、视频是示教来源；DAPG 用示教预训练并加增广损失，真实机器人上训练时间由 4–7 小时降到 2–3 小时（引 Zhu 等）；IRL 在手内操作中用得很少；GAIL 在灵巧手上优于 BC 与直接 RL。
- **仿真与数据集（§II-F/G）：** MuJoCo、Isaac Gym 支撑大规模训练，但摩擦、软体形变和接触力建模仍有 sim-to-real 缺口；欠驱动手的仿真尤其困难。
- **开放问题（§VI）：** 数据效率、sim-to-real、软体手、触觉感知、示教采集平台、任务泛化。

## 对 wiki 的映射

- [`wiki/concepts/dexterous-kinematics.md`](../../wiki/concepts/dexterous-kinematics.md)
- [`wiki/methods/in-hand-reorientation.md`](../../wiki/methods/in-hand-reorientation.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2401.07915>
