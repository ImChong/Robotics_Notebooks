# PingpongDuo: Agile Humanoid Table Tennis Doubles with Safe Team Coordination（arXiv:2610.20579）

> 来源归档（ingest）

- **标题：** PingpongDuo: Agile Humanoid Table Tennis Doubles with Safe Team Coordination
- **arXiv：** <https://arxiv.org/abs/2610.20579>（v1 提交 2026-10-14；v2 2026-11-13；PDF：<https://arxiv.org/pdf/2610.20579>）
- **项目页：** <https://www.archon.tech/research/PingpongDuo>（页内提供论文 PDF、完整视频、交互式仿真回放与 BibTeX；**无代码链接**）
- **作者：** Haoran Jiang, Yichao Zhong, Kai Zhang, Cheng Liu, Yixuan Pan, Zhenjie Yang, Jingbo Wang, Tianyu Li, Li Chen
- **机构（PDF 首页）：** 上海创智学院（Shanghai Innovation Institute）；源策未来（Archon Robotics）；香港大学（HKU）；英伟达（NVIDIA）
- **入库日期：** 2026-10-15
- **再核日期：** 2026-11-21
- **核对范围：** 通读 `https://arxiv.org/html/2610.20579`（v2，2026-11-13）全文（Abstract、§I 引言、§II 相关工作、§III 系统总览、§IV 风险感知团队协同、§V 步法/击球技能与球拍臂残差、§VI 实验与 Table I–III、§VII 结论与局限），并复核项目页 `<https://www.archon.tech/research/PingpongDuo>` 确认无代码仓库链接。边界：未在仿真/真机复现。
- **通读：** 已完成
- **类型：** paper / humanoid / table-tennis / multi-robot-coordination / rl / sim2real
- **一句话说明：** 双台 Unitree G1 乒乓球双打：团队层联合选择站位并用学习风险估计过滤提案，个体层分离步法与击球技能，再用球拍臂残差在标定仿真中做回球反馈修正。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-15）：** 项目页按钮为 Paper（PDF）、Full video、Simulation（网页内交互回放）与 BibTeX；页面中唯一的 github.com 链接是作者个人主页，不是代码仓库。
- **论文核查：** PDF 中未出现 "code will be released" 或代码仓库声明。
- **结论：** **未开源**（截至 2026-10-15，项目页与论文均未列代码或权重）。

## 摘录 1：问题与主张（Abstract / §I）

- **痛点：** 双打要求两台机器人在狭窄工作区内严格交替击球；快速重定位会损伤击球精度，队友之间不协调则有碰撞风险。
- **方法：** 团队层规划器联合选择站位目标，使用学习到的风险估计过滤危险提案，并优先保留当前击球者的目标；个体层分为步法技能与击球技能；击球后再由球拍臂残差修正。
- **贡献声明（§I）：** (i) 面向双打的分层框架，把带风险感知的联合站位与步法/击球技能结合；(ii) 击球残差策略，用仿真回球反馈修正击球后的动作。

## 摘录 2：仿真与真机要点（摘要 / §VI，数字取自论文）

| 发现 | 要点 |
|------|------|
| 摘要：仿真回球成功率 | 相对 SOTA 基线 **+13.2%**（摘要未区分相对/绝对提升，未核实） |
| 摘要：清球定位误差 | 约 **2 cm** |
| 摘要：真机清球-回位 | 2.15 m 清球与回位 **2.4 s** 完成，峰值 **2.23 m/s** |
| 摘要：真机双机交替 | 两台机器人连续 **10+** 次严格交替击球 |
| Table I（规划器对比，仿真） | RL + Learned Risk（本文）平均回报 **4.87±0.74**；Rule-based **3.60±0.40**；Rule-based + Heuristic CBF / + Learned Risk 约 **3.79** |
| Table III（策略对比，仿真，5 seeds） | 本文 Composed：SRC **67.4%**、SRH **55.6%**、Ep **7.43 cm**；复现的 HITTER Composed：SRC 55.2%、SRH 42.4%；复现的 SMASH Composed：SRC 53.8%、SRH 35.8% |
| 消融：击球残差（§VI-D） | 去掉 arm refinement 时 SRC 55.2% → 67.4%，SRH 44.2% → 55.6% |

- Table II 已明确给出指标定义：$SR_C$ 为球拍触球率（Percentage of balls contacted by the racket），$SR_H$ 为合法回球成功率（Percentage of balls legally returned），$E_p / E_v / E_n$ 分别为击球时刻 $t^\star$ 的球拍位置/速度/法向误差，$E_M$ 为让位结束横向位置误差，MPJPE 为骨盆对齐后的全身参考跟踪误差。
- 局限（§VII）：真机依赖外部 MoCap 估计球与机器人状态，尚未包含球旋转测量与空气/碰撞自旋建模。
- 基线 HITTER [4] 与 SMASH [9] 为作者按双打移动指令改编后复现，不是原作者实现。

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-pingpongduo.md`](../../wiki/entities/paper-pingpongduo.md)：团队层-个体层分层图；强调「站位-击球」耦合与风险过滤；标注开源状态为未开源。
- 与 [Archon Robotics 公司页](../../wiki/entities/archon-robotics.md)、[乒乓球分层技能学习](../../wiki/methods/table-tennis-strategy-skill-learning.md)、[人形多机协同](../../wiki/concepts/humanoid-multi-robot-coordination.md) 互链。
