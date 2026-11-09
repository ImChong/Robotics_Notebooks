# Dual-Arm Tabletop Rearrangement with Movable Stacking and Gap-Aware Push Transfers（arXiv:2611.07726）

> 来源归档（paper · 原文通读，支撑 `wiki/entities/paper-dual-arm-movable-stacking-gap-aware-push.md`）

- **标题：** Dual-Arm Tabletop Rearrangement with Movable Stacking and Gap-Aware Push Transfers
- **作者 / 机构：** Mohammadreza Firoozalizadeh、Zakiye Rostamirad、Arman Barghi、Ahmad Kalhor、Mohammadreza Nayeri、Mehdi Tale Masouleh；Human and Robot Interaction Laboratory (TaarLab)，School of Electrical and Computer Engineering，University of Tehran，Tehran, Iran（首页署名）
- **类型：** paper / dual-arm / tabletop-rearrangement / MCTS
- **arXiv：** <https://arxiv.org/abs/2611.07726>（v1，cs.RO，2026-11-05）
- **PDF：** <https://arxiv.org/pdf/2611.07726v1>（6 页；pdftotext 全文 + 逐页看图 Table I/II、Fig. 3/5/6）
- **发表信息：** 首页脚注：accepted author manuscript，Proc. 34th International Conference on Electrical Engineering (ICEE), 2026，DOI 10.1109/ICEE70628.2026.11710769
- **代码：** 论文未给出代码或项目页链接
- **一句话说明：** 在双臂桌面重排中，用「间隙感知推动转移」替代固定交接位姿做跨边界物体交换，并与可移动堆叠一起放进同一个联合 MCTS 规划器；仿真基准上物体越多优势越明显。
- **核对范围：** 通读 6 页全文：摘要、§I 引言、§II-A 问题形式化、§II-B 规划与边界转移、§III 结果与讨论（含 Table II 全部 16 行数字）、§IV 结论、参考文献列表；浏览 Fig. 1–7 的版面与图注（Fig. 7 柱状图与置信区间只看了图注与正文描述，未从图中读数）。未阅读被引文献 [4] Barghi 等（arXiv:2509.22828，可移动堆叠）与 [7] Gao & Yu（IROS 2022，固定交接基线）的原文，相关表述仅依赖本文转述。论文无附录 / 补充材料。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题设定（§II-A）：** 物体集合 O 为带矩形足迹 (h_i, w_i)、中心 (r_i, c_i) 的对象，桌面 H×W；两台固定基座机械臂 r0、r1 分管相邻两半工作空间，共享边界列 c_b=⌊W/2⌋ 是转移接口。沿用 [4]，环境含 12 类异质物体，按支撑能力分为主基座（box、pot、basket）、次基座（bowl、mug、paper cup）、轻物（spoon、fork、knife）、重物（apple、pear、banana）。状态 s_t=(P_t, R_t, m^0_t, m^1_t)，R_t[i,j]=1 表示 i 直接叠在 j 上，m 为末端位置。
- **动作集（§II-A）：** A={move, stack, push, wait}；move 为抓取式放置，stack 放在有效支撑物上，push 仅用于跨边界转移，wait 在对臂无安全并行动作时使用。联合动作 a_t=(a^0_t, a^1_t)；不得同时操作同一源物体，并发 stack 不得指向同一支撑物，其余运动干涉 / 顺序依赖时一臂动作配 wait。
- **可移动堆叠（§II-A）：** 搬基座物体时，其上稳定承载的物体连同相对堆叠关系整体搬运（来自 [4]），减少多余中间重排。
- **联合 MCTS（§II-B）：** 节点为状态、边为可行联合动作；选择用 UCB；扩展对未解决或阻碍目标的物体生成单臂候选，再组合成满足协调约束的联合动作，优先兼容并行；评估按「已到达目标位置 / 目标支撑关系的物体数」打分，另加「暴露更多直接可做动作」的小偏好；回传更新。规划器是「进度驱动」而非显式最小化代价。
- **间隙感知推动转移（§II-B，Fig. 5–6）：** 选定推动转移后，在共享界面两侧的边界带内找可行穿越区间，仅保留对物体足迹与推前对准动作有足够间隙者；候选行集合 G(o_i) 上选 r_i^cross = argmin_ρ (|ρ−r_i| + |ρ−r_i^*|)（r_i 当前行，r_i^* 目标行，无关时退化为最近当前行）。执行三阶段：对准移动到穿越行附近，直线推过边界，随后接收臂可取用。穿越位置是规划变量而非预设。
- **执行代价（§II-B）：** J(Π)=Σ C(a_t)；move/stack 含机械臂到物体的行程、搬运到放置点与固定原语项；push 含接近、对准、直推、撤回。仅在规划后评估，不参与搜索。
- **基准设置（§III）：** 桌面 1×2 m，共享边界在中间列；N∈{5,…,12}；每个 N 生成 20 对随机初始 / 目标场景，每种方法每对跑 3 次；物体尺寸异质，沿用 [4] 的设置；基线为用同一 MCTS 骨架实现的 Fixed Handoff（把跨边界交换限制在单一预设转移位姿，来自 [7]）；指标 Joint Steps（协调执行步数）与 Execution Cost。
- **Table II（Fixed Handoff → Proposed）：** Joint Steps：N=5 6.67→6.60，6 8.00→7.98，7 10.58→9.93，8 12.26→11.38，9 12.90→12.48，10 13.77→12.40，11 16.11→14.83，12 17.52→15.72。Execution Cost：N=5 9.63→9.96，6 11.73→12.47，7 16.15→15.96，8 20.25→18.99，9 20.63→20.02，10 22.14→21.30，11 26.90→25.64，12 29.48→27.99。
- **作者在正文的百分比（§III）：** N=5 Joint Steps 降约 1.0%，Execution Cost 升约 3.4%；N=6 Joint Steps 仅差 0.02，Fixed Handoff 的 Execution Cost 仍更低；N=7 起两项均更低，N=7 步数降约 6.1%、代价降约 1.2%；N=12 步数降约 10.3%、代价降约 5.1%。
- **作者结论（§III–§IV）：** 两方法同骨架、同堆叠能力，差异来自转移机制；场景相关的穿越位置减少了密集场景的绕行；Fixed Handoff 在稀疏场景仍有竞争力，场景越密提出方法优势越明显。

## 作者自述 vs 本页判断

- 作者自述：上面「核心摘录」与「作者结论」。
- 判断（非论文原话）：评测是抽象网格级任务规划比较，未见真机或物理仿真、规划耗时、成功率、置信区间数值与显著性检验；N=5、6 的 Execution Cost 反而更差；「异步」收益只通过 Joint Steps / Execution Cost 间接体现，未单独量化「减少等待 / 同步」。
- 引言（§I）称 Barghi 等 [4] 为单臂方法（Table I），本文是其向双臂的扩展（推测：同实验室延续工作，作者重合）。

## 对 wiki 的映射

- [paper-dual-arm-movable-stacking-gap-aware-push](../../wiki/entities/paper-dual-arm-movable-stacking-gap-aware-push.md)
