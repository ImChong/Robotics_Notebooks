# REGRIND: A Minimalist Retargeting-Guided Reinforcement Learning Recipe for Dexterous Manipulation

> 来源归档（ingest · 复核 2026-09-09）

- **标题：** REGRIND: A Minimalist Retargeting-Guided Reinforcement Learning Recipe for Dexterous Manipulation
- **缩写：** **REGRIND** / REtargeting-Guided ReINforcement learning for Dexterous manipulation
- **类型：** paper / dexterous-manipulation / motion-retargeting / reinforcement-learning / sim2real / contact-rich-manipulation
- **arXiv：** <https://arxiv.org/abs/2607.11874>（PDF: <https://arxiv.org/pdf/2607.11874>；HTML: <https://arxiv.org/html/2607.11874>）
- **项目页：** <https://www.yunhaifeng.com/REGRIND/>
- **代码：** <https://github.com/yunhaif/regrind>
- **机构：** 康奈尔大学（Cornell University）；亚马逊 FAR（Amazon FAR / Frontier AI & Robotics）
- **作者：** Yunhai Feng, Natalie Leung, Jiaxuan Wang（Cornell）；Lujie Yang, Haozhi Qi（Amazon FAR）；Preston Culbertson（Cornell）
- **状态：** arXiv 预印本（2026）；**已开源**（MIT 代码 + 预计算重定向轨迹）
- **入库日期：** 2026-07-16（初入库）；**复核：** 2026-09-09
- **一句话说明：** 单次光学 MoCap 人手–物体演示 → **interaction mesh 交互保留重定向**（OmniRetarget 同族）→ **残差 RL** 跟踪物体关键点 + RSI/SE(3) 增广 → 系统辨识后零样本部署 **LEAP/WUJI** 剪刀与螺丝刀；系统实验总结 contact-rich 灵巧 sim2real 关键因素。

## 摘录 1：问题与核心主张

- **动机：** 人形 WBT 已验证「重定向参考 + RL 跟踪」极简配方；灵巧 **contact-rich manipulation** 需精细调节接触模式与力，纯运动学重定向易 **穿透、丢失接触结构**，下游 RL 与 sim2real 难。
- **REGRIND 管线：** 单次人类演示 → **保留 hand–object 空间/接触关系** 的机器人参考 → 仿真 **残差 RL** 跟踪 **物体-centric 关键点** → **系统辨识** 后 **零样本** 真机。
- **真机展示：** LEAP / WUJI 双手 × 剪刀 / 螺丝刀；流体、类人行为；项目页含 sim vs real 1× 与初态泛化 2× 视频。

**对 wiki 的映射：** [`wiki/methods/regrind-retargeting-guided-rl.md`](../../wiki/methods/regrind-retargeting-guided-rl.md)

## 摘录 2：交互保留重定向

- **输入：** MANO 手部关键点 + 物体 6D（铰接含关节角）；剪刀来自 ARCTIC，螺丝刀自采光学 mocap。
- **优化：** 物体 + 人手语义关键点 Delaunay **interaction mesh**；最小化源/机器人 mesh **Laplacian 坐标差** + 时序平滑；逐帧 SQP（**Drake + MOSEK**，可换 Clarabel）。
- **继承：** [OmniRetarget](https://arxiv.org/abs/2606.16272) formulation；相对 DexMachina functional retargeting **显式保留交互语义**。

**对 wiki 的映射：** 方法页「主要技术路线 / 流程总览」；对照 [TopoRetarget](../../wiki/methods/toporetarget-interaction-preserving-dexterous-retargeting.md)。

## 摘录 3：残差 RL + 增广 + Sim2Real

- **控制：** 残差叠在名义参考关节 $\bar{q}_t$ 上 → PD；$q^{\text{target}}=\bar{q}_t+\alpha\odot\pi_\theta$。
- **奖励：** **物体-centric 关键点** 指数距离跟踪（无需显式接触先验）。
- **探索：** **RSI** 从重定向轨迹采样 episode 初态；训练时 **SE(3) 增广**（物体初态 ±5 cm / ±30° 时对整条参考插值 warp，**不重解**重定向）。
- **Sim2Real：** DR（摩擦、增益、时延）、观测噪声、推力/重力课程；部署 **MoCap 馈物体位姿**（隔离感知）；**系统辨识** 不可省略。

**对 wiki 的映射：** 方法页「实验要点 / 工程实践 / 结论」。

## 摘录 4：实验数字（Table 1–3）

| 设置 | REGRIND | 对照印象 |
|------|---------|----------|
| 仿真四任务 SR（1024 rollout） | **98.7–99.8%**；关键点误差 **5.3–6.5 mm** | DexMachina scissors **0–22%**；Mink IK+RL **0–3%** |
| SPIDER 轨迹 + residual RL | SR **0%** | MPC 轨迹偏离演示，不适合 residual 初始化 |
| 真机（演示初态） | LEAP 剪刀 **9/10**、螺丝刀 **10/10**；WUJI 螺丝刀 **9/10** | WUJI-Scissors **0/10** |
| 初态泛化 ±5 cm / ±30° | 与演示初态性能接近（Table 3） | 增广在 RL 训练时动态生成 |

**对 wiki 的映射：** 方法页实验表与结论。

## 摘录 5：开源边界（步骤 2.5 · 2026-09-09）

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — [yunhaif/regrind](https://github.com/yunhaif/regrind)（MIT，~100★） |
| **预计算轨迹** | 仓内已附带 retargeted `.h5`，可 **跳过重定向** 直接 RL |
| **重定向依赖** | 可选：`pydrake` + **MOSEK** license（或 `solver=clarabel`） |
| **仿真栈** | **Isaac Sim 5.1.0** + **Isaac Lab 2.3.0** + `rsl_rl` |
| **训练/评测脚本** | `scripts/rsl_rl/train.py`、`play.py`；任务 `Regrind-{LeapHand,WujiHand}-{Scissors,Screwdriver}-v0` |
| **未开源** | 无；论文下一步为 **vision-based** 蒸馏（部署仍依赖 MoCap 物体状态） |

**对 wiki 的映射：** [`sources/repos/regrind.md`](../repos/regrind.md)、[`sources/sites/regrind-project-yunhaifeng.md`](../sites/regrind-project-yunhaifeng.md)
