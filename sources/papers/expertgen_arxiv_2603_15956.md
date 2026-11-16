# ExpertGen（arXiv:2603.15956）

> 来源归档（paper）

- **标题：** ExpertGen: Scalable Sim-to-Real Expert Policy Learning from Imperfect Behavior Priors
- **类型：** paper
- **作者：** Zifan Xu, Ran Gong, Maria Vittoria Minniti, Ahmet Salih Gundogdu, Eric Rosen, Kausik Sivakumar, Riedana Yan, Zixing Wang, Di Deng, Peter Stone, Xiaohan Zhang（共同指导）, Karl Schmeckpeper（共同指导）；机构：Robotics and AI Institute、UT Austin、Sony AI
- **arXiv：** <https://arxiv.org/abs/2603.15956>（v1，2026-03-16，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2603.15956v1>
- **项目页：** <https://bdaiinstitute.github.io/expertgen>（论文首页所列；抓取时未见仓库链接，故不登记代码仓库）
- **入库日期：** 2026-11-12
- **一句话说明：** 用少量不完美示范（LLM 脚本或人类遥操）训练状态扩散策略作行为先验，在并行仿真里用 FastTD3 只优化扩散模型的初始噪声（DSRL），仅靠稀疏成功奖励得到专家策略，再经 DAgger 蒸馏成视觉策略零样本上真机。
- **核对范围：** 下载 arXiv v1 PDF（21 页），pdftotext 后通读正文 I–VIII 全部章节、公式（1）–（14）、表 I–V 与图 1–8 的文字与图注，并通读附录 A–C（超参表 VI–IX、基准与状态/动作定义表 X–XI、表 XII–XV、脚本策略代码图 13–21）；表 IV 用页面截图核对。未做：参考文献逐条核对、附录里脚本策略代码的逐行审读（只看了结构）、图 6–8 与图 11–12 的柱状/曲线图读数（只引用正文与表中数字）、项目页视频与代码（未见仓库）。
- **通读：** 已完成

## 核心摘录（均为论文自述，章节号为论文章节）

1. **流程（Sec. IV，图 1/2）：** 三阶段。阶段 1 用不完美示范训练状态条件扩散策略作行为先验；阶段 2 用 DSRL 在 IsaacLab 并行仿真里只学扩散模型的初始噪声，底层扩散策略冻结，RL 算法用 FastTD3；阶段 3 用 DAgger 把状态专家蒸馏成点云或 RGB 视觉策略并加视觉域随机化。
2. **奖励（Sec. III-A）：** 二值成功指示 r = I_succ，无奖励工程；约束 MDP 里的可行集 F 未显式给出。
3. **示范来源（Sec. V-A）：** AnyTask 8 个桌面任务，LLM 合成脚本策略各采 1000 条；AutoMate 为反转"拆卸"轨迹近似装配（动力学不对称导致 mismatch），各采 500 条。
4. **仿真结果（Sec. VI-A，表 I，图 3）：** AnyTask 8 任务上 ExpertGen 为 99.8 / 99.7 / 99.3 / 100 / 83.3 / 67.2 / 80.7 / 52.1（%，1024 条 rollout，Wilson 95% 区间）；AutoMate 平均 90.5%（256 条 rollout）。摘要写"长程任务 85% overall"，与表 I 八任务算术平均 85.3 相符，但该平均含四个短程任务。
5. **失败恢复（Sec. VI-B，表 III）：** 随机张开夹爪与随机外力扰动下，Lift Banana 外力成功率 ExpertGen 56.6% 对基础扩散策略 1.9%。
6. **平滑度（Sec. VI-C，表 II）：** 平均 jerk，ExpertGen 5.97、Residual RL(0.1) 35.3、Residual RL(0.02) 4.65、扩散策略 7.31；DTW 以扩散策略最低。
7. **真机（Sec. VI-D，表 IV/XII）：** Franka + Robotiq 2F-85 + 3 个 RealSense D435i；点云学生 Lift Banana / Push Pear / Open Drawer 为 75.0 / 65.0 / 85.0%，对照 AnyTask BC 为 73.3 / 16.7 / 42.5%，Lift Peach 80.0% 对 62.1%（表 XII）；RGB（108×192）Lift Banana 正文称 80% 对 BC 0%。DAgger 的 α 退火表 V 显示低而非零的最终 α 最优（75.0%）。
8. **分析（Sec. VII）：** 约 200 条示范即够，降到 50 条平均成功率降到 58.2%（图 7）；FastTD3 优于 PPO（表 I 的 ExpertGen-PPO 行）；Stack Banana on Can 上人类示范经 SkillMimicGen 扩增后 ExpertGen 由 67.2% 升到 79.2%（表 XV，6 条人类示范扩到 1000 条）。
9. **作者承认的局限（Sec. VIII）：** 专家被先验覆盖范围限制，缺少的技能（如把倒下的肉罐头扶正）学不出来。

## 核对中发现的论文内部不一致（我的判断，非作者自述）

- 表 IV 的 RGB 列把 80% 放在 AnyTask 行、ExpertGen 行为"–"，而 Sec. VI-D(b) 正文说 ExpertGen 蒸馏策略 80%、BC 为 0%；表 XII 里 Lift Peach 一列才是 80.0 对 62.1。表 IV 的 RGB 列大概率行标有误（推测）。
- Sec. VI-A 正文称 AutoMate 为"八个"任务，附录 B/图 3/图 10 为九个资产（含 00726）。
- Sec. VI-B 称外力扰动平均下降 28.6%，按表 III 三个任务的 43.1 / 0.2 / 45.3 算术平均为 29.5（可能是取整或权重差异，推测）。

## 对 wiki 的映射

- [paper-expertgen](../../wiki/entities/paper-expertgen.md)
