# PriMoRetarget 论文来源归档

## 元数据

- **题目：** PriMoRetarget: Humanoid Motion Transfer Through Prioritized Motion Retargeting
- **作者：** Nicholas Ioannidis, Dun Yang, Max Smith, Tue Vu, Ziyu Zhang, Shayegan Omidshafiei, Xue Bin Peng
- **单位：** FieldAI（前五位与 Omidshafiei）；Simon Fraser University（Ziyu Zhang、Xue Bin Peng；Peng 另列 NVIDIA 与 Alberta Machine Intelligence Institute）
- **预印本：** https://arxiv.org/abs/2610.12642
- **HTML：** https://arxiv.org/html/2610.12642v1
- **版本：** v1，2026-10-08，cs.RO / cs.GR
- **项目页：** https://nickioan.github.io/primoretarget-page/ （归档见 sites/primoretarget-project.md）
- **代码：** 论文称发表后公开代码与重定向数据；项目页写 “Code (coming soon)”，无仓库链接。
- **读取范围：** arXiv 摘要页与 HTML 全文抽取；各数据集片段数文中未给。

## 论文要解决的问题

运动重定向把人类动作转到人形机器人上。已有方法在单角色上较好，但多角色/物体交互引入必须保持的接触与空间关系。常规做法是加权多目标优化，需要仔细调权重且仍可能损害关键运动质量。

## 方法与训练信息

1. **思路：** 用显式优先级层级替换相对权重；低优先级更新被约束为保持高优先级目标（脚落点、交互接触优先于一般姿态目标）。
2. **形式：** 词典序、任务优先级风格——每一级在所有更高级的解集内优化（Eq. 2–3），用一阶零空间约束局部实施；阻尼零空间投影 P = I − Jᵀ(JJᵀ + λI)⁻¹J。
3. **效率：** 每帧可分离使雅可比按时间分块对角，复杂度从 O(T³m³) 降到 O(Tm³)（m 为每帧更高优先级误差分量数）。
4. **求解器：** 普通梯度下降（x ← x − α·g_total），未指定专用 IK 求解器；先有一小段统一加权的预热阶段再施加层级。
5. **层级：** (1) 脚位置匹配，(2) 接触保持，(3) 其余姿态与交互目标。时间平滑、关节限位、地面穿透作为不分级的“自由目标”直接加入梯度。
6. **状态：** 每帧 d = 6 + nθ（根位置、根朝向、关节角）。
7. **数据与机器人：** LAFAN1（单角色）、OMOMO 大箱子类（角色–物体）、Inter-X 辅助起身子集（角色–角色）、自有 35 分钟多角色协作操作数据；主机器人 Unitree G1，另测 Deep Robotics DR02、Unitree H1-2、Booster T1。
8. **下游跟踪：** 每条动作单独训练 PPO 策略（DeepMimic 风格），各方法超参固定，输出关节 PD 目标，G1 仿真，共 50 条动作；成功判据为机器人或物体偏差不超过 0.5 m。

## 论文报告的结果

- **运动学指标**（穿透时长% / 穿透深度 cm / 脚滑 cm/s / 悬浮 cm / 抖动 m/s³ / 接触保持%）：
  - LAFAN1：GMR 12.1/10.7/5.2/2.1/216.7；OmniRetarget 1.5/0.7/9.4/0.6/229.4；Ours 12.2/3.9/3.5/1.4/135.6。
  - OMOMO：PHC 94.1/9.3/12.9/0.2/76.3/63.2；GMR 65.4/5.1/1.7/0.1/94.7/48.6；OmniRetarget 21.6/2.7/4.9/0.3/205.2/86.4；Ours 13.4/1.8/3.7/2.4/108.0/96.9。
  - Inter-X：PHC 79.0/9.7/8.8/2.8/84.0/35.2；GMR 67.8/7.1/3.9/1.3/147.2/26.4；OmniRetarget 33.6/5.7/3.4/0.2/136.7/26.9；Ours 55.7/6.9/2.4/1.8/103.9/78.8。
  - 自有数据：GMR 39.3/6.2/4.4/5.1/102.3/29.3；OmniRetarget 10.2/2.0/2.7/0.1/129.6/26.5；Ours 26.4/5.6/1.7/3.2/67.5/92.4。
- **RL 跟踪（成功率% / 身体误差 cm / 物体误差 cm）：**
  - LAFAN1：GMR 90.0/4.0；OmniRetarget 90.0/5.2；Ours 100.0/1.8。
  - OMOMO：PHC 86.6/7.4/31.1；GMR 79.8/7.9/38.7；OmniRetarget 99.1/3.5/5.8；Ours 100.0/2.8/3.0。
  - Inter-X：PHC 72.8/8.7；GMR 57.2/15.9；OmniRetarget 66.1/12.1；Ours 93.6/4.5。
  - 自有数据：GMR 24.1/30.9/137.1；OmniRetarget 4.3/36.8/166.4；Ours 62.6/12.5/47.4。
- **消融（成功率%）：** 去预热时自有数据 78.5、Inter-X 70.3；“Free-last”（自由目标放最后一级）自有数据 92.3、Inter-X 80.8。这些数字与主表中完整方法的自有数据 62.6% 不在同一口径（消融数值反而更高），原文设置本次未能厘清，不据此下结论。
- **真机：** 选定动作部署于 G1，含两台机器人协作搬运（Fig. 6，无定量成功率）。

## 复现与解释边界

- 论文自述：保持源运动结构，故源数据的抖动、穿透、采集环境不一致会传到输出；物体轨迹大体沿用源数据，限制对目标本体的适配；优先级层级需手工设定；多机器人协作操作仍难（自有数据 RL 成功率 62.6%）。
- 运动学指标并非全面占优：LAFAN1 穿透时长 12.2% 与 GMR 相近，OmniRetarget 穿透更低（1.5%）；OMOMO 悬浮 2.4 cm 高于其他方法。
- 每条动作单独训练 PPO，不是通用跟踪策略的结果。
- 代码与数据尚未公开。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-primoretarget.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md
- 概念入口：wiki/concepts/motion-retargeting.md
- 方法入口：wiki/methods/motion-retargeting-gmr.md

## 一手来源

- arXiv：<https://arxiv.org/abs/2610.12642>
- arXiv HTML：<https://arxiv.org/html/2610.12642v1>
- 项目页：<https://nickioan.github.io/primoretarget-page/>
