# Rapid On-Robot Learning for Dynamic Manipulation Skills: Robot Juggling

> 来源归档（ingest · arXiv HTML）

- **标题：** Rapid On-Robot Learning for Dynamic Manipulation Skills: Robot Juggling
- **类型：** paper / dynamic-manipulation / on-robot-learning / contact-rich / bimanual / juggling / memory-based-learning
- **arXiv abs：** <https://arxiv.org/abs/2608.26800>
- **arXiv HTML：** <https://arxiv.org/html/2608.26800v1>
- **PDF：** <https://arxiv.org/pdf/2608.26800>
- **项目页：** **无独立 juggling 项目页**；硬件背景见 RAI 博客 [`sources/sites/rai-athenazero-blog.md`](../sites/rai-athenazero-blog.md)
- **代码 / 数据：** 截至 2026-09-05 **未列** GitHub / Hugging Face / 数据集
- **作者：** Taeyoon Lee\*、Chunpeng Wang、Christopher G. Atkeson、Alfred A. Rizzi、Nicolas Rojas（\* 通讯：tlee@rai-inst.com）
- **机构：** 机器人与人工智能研究所（RAI Institute）；卡内基梅隆大学（CMU，Atkeson）
- **入库日期：** 2026-09-05
- **一句话说明：** 正则化记忆学习 + 互达集（MRS）安全约束，让 AthenaZero 双臂多指手在 **<5 分钟** 真机交互内学会五种三球抛接花样（cascade / tennis / half-shower / shower / box）；先验模型零样本连一轮都完不成，但其一阶信息仍能引导学习。

## 开源状态（步骤 2.5）

- **核查日：** 2026-09-05，打开 <https://arxiv.org/html/2608.26800v1> 与 <https://rai-inst.com/resources/blog/bimanual-robot-for-dynamic-manipulation/>。
- **已发布：** arXiv 全文 + 补充视频（Movie 1、S1–S4）；RAI 博客介绍 AthenaZero 硬件与棒球任务。
- **未发布：** 官方 GitHub、权重、数据集、独立项目页 Code 按钮。
- **结论：** **确认未开源**。wiki `## 源码运行时序图` 标 **不适用**。

## 摘录 1：不完美先验 + 正则化记忆学习

灵巧抛接的核心矛盾是：仿真/解析先验与真机接触（摩擦、柔顺球、多指释放时序）差距大，零样本连 **一轮** 三球 cascade 都完不成；但先验的一阶梯度仍告诉学习者「命令怎么改可能有效」。框架用 **正则化记忆学习**：每次抛球把 \((x_i,u_i,y_i)\) 写入记忆，在 **k-NN + RBF 权重** 的局部线性回归里，用 Frobenius 正则把局部雅可比 \((C,D,d)\) 拉向先验 \(\partial f_0/\partial x,\partial f_0/\partial u,f_0\)。抛球任务里先验是 **恒等映射** \(f_0(x,u)=u\)（命令=期望落点），\(\gamma=0.001\) 在「信经验」与「信先验」间折中。学习在 **抛接进行中实时发生**，不是只在 reset 间 episodic 更新。对比 Ploeger & Peters（杯式 open-loop 正则记忆）：本文在 **可复用闭环技能** 上学习，经验可跨花样与不同入态 \(x\) 复用。

**对 wiki 的映射：**
- [paper-robot-juggling-athenazero](../../wiki/entities/paper-robot-juggling-athenazero.md)
- [sim2real](../../wiki/concepts/sim2real.md) — 不是 zero-shot 迁策略，而是「先验错得离谱仍能当学习正则」
- [contact-rich-manipulation](../../wiki/concepts/contact-rich-manipulation.md)

## 摘录 2：互达集 MRS 让高风险动态练习可重复

单独满足关节位/速/加速度限的抛球，可能把臂带到 **下一接抛不可达或不可延续** 的状态（89% 无 MRS 的 task-level 解被判 unsafe）。**Mutually Reachable Set** \(\mathcal{M}_S\) 离线用 B-spline 前向/后向可达集交构造凸多面体，在线 task planner 强制 \(\mathbf{M}\mathbf{s}_f\leq\mathbf{m}\)。低层仍用 **Ruckig** 1 kHz 时间最优 jerk 轨迹 + 逆动力学跟踪。这样机器人可以在接近物理极限的动态动作上反复练习，而不必把速度/workspace 收得过保守。消融：有 MRS 约束的 7578 次 planner 查询 **100% safe by construction**。

**对 wiki 的映射：**
- [paper-robot-juggling-athenazero](../../wiki/entities/paper-robot-juggling-athenazero.md)
- [model-predictive-control](../../wiki/methods/model-predictive-control.md) — 任务级规划 + 硬安全集
- [whole-body-control](../../wiki/concepts/whole-body-control.md)

## 摘录 3：AthenaZero 真机栈与五种三球花样

**AthenaZero**（RAI 定制低惯量双臂：1-DoF 躯干 + 双 7-DoF 臂 + 双 6-DoF 欠驱动三指手，Bowden 拉线 ~80 ms 开合）。感知：躯干顶 **Lucid Helios2 ToF + Triton RGB**（30 Hz，端到端延迟 ~0.1 s）；少量点标注即可配置新物体跟踪。控制：**ROS2 Humble** 多节点异步；Orchestrator 按墙钟调度 throw/catch 技能；抛球走学习器，接球闭环重规划至 catch 前 0.1 s。球：130 g 软豆袋（HB Juggling #1003）。**五种三球花样** cascade / tennis / half-shower / shower / box，可切换。学习数据（不含人工捡球 reset 时间）：cascade 平均 **53 s** 物理交互、tennis→half-shower→cascade 序列 **75 s**；shower/box 约 **30 s / 60 s** 到平台期。五次独立 trial 均在第 8 次 reset 前学会 cascade 并连续成功 3 次。侧抛 shower/box 因飞行时间短于感知延迟，接球只能 **开环定点**，成功率低于 cascade 类。

**对 wiki 的映射：**
- [paper-robot-juggling-athenazero](../../wiki/entities/paper-robot-juggling-athenazero.md)
- [manipulation](../../wiki/tasks/manipulation.md)
- [paper-spd](../../wiki/entities/paper-spd.md) — 对照：SPD 用仿真 VR 预训练再短微调；本文是真机分钟级在线记忆学习
- [paper-smpc2rl-loco-manipulation](../../wiki/entities/paper-smpc2rl-loco-manipulation.md) — 同 RAI 研究所，不同任务（全身 loco-manip vs 原地动态操作）

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 关联 wiki 页面的参考来源段落已添加 ingest 链接
