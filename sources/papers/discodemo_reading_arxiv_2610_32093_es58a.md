# DiscoDemo 论文通读（成功示范多样性视角）

> 来源归档（原文通读 · 支撑 `wiki/overview/embodied-station-2026-10-29-58-papers-technology-map.md`）

- **标题：** DiscoDemo: Discovering Efficient and Diverse Robot Demonstrations for Imitation Learning
- **作者：** Minho Park, Kinam Kim 等（KAIST / Holiday Robotics；通讯 Hojoon Lee、Jaegul Choo）
- **类型：** paper / imitation-learning / data-generation
- **原文：** <https://arxiv.org/abs/2610.32093>（2026-11 通过 arXiv PDF v1 获取全文，pdftotext 转文本）
- **一句话说明：** 把强化学习当作示范生成器而非部署控制器：并行逆向课程 + 成功门控的 METRA 式多样性奖励 + 安全塑形/过滤，生成多种成功解法的静态数据集，再微调 π0.5 做模仿学习。
- **核对范围：** 通读摘要、§1–§4（含 RQ1–RQ4 实验、结论与局限）及附录 A（任务/仿真设置）、B（生成器实现、逆向课程、多样性目标、安全项）、C（基线实现）、D–F 的方法与指标定义；附录 F 的逐任务补充曲线/表与参考文献只浏览数字量级，未逐项核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题与前提（§1）：** BC 对数据覆盖敏感，遥操作成本高；奖励驱动的 RL 容易收敛到少数解，滚出更多轨迹也不增加状态覆盖。前提：多种成功解法的并集比单一窄解覆盖更广。
- **三阶段流程（图 2、§3）：** (a) 特权状态的潜变量条件生成器 πGen 用并行逆向课程 + 稀疏任务奖励 + 成功门控多样性 + 安全塑形训练；(b) 冻结后以 z~N(0,I) 采样生成并做成功与安全过滤；(c) 导出 (观测, 动作) 静态数据集，训练仅用视觉 + 本体的下游策略 πIL，不再调用生成器。
- **逆向课程（式 1、§B.2）：** 从人类示范末段状态重置，成功率达阈值 p（附录为 0.9，最近 10 个回合）后把边界 c_i 往前移 Δc；一半示范解到 c=0 后切回原始分布 ρ0；约 2048 个并行环境、每条示范约 100 个。用 FlashSAC 训练 5000 万步（单 B200）。
- **多样性（式 2–3）：** r_div=⟨φψ(s')−φψ(s), z⟩（METRA）；r̃=r_task+α·y(τ)·r_div，y 为轨迹成功指示，α 主实验取 0.3；去掉门控则多样性奖励压过稀疏任务奖励、生成器学不会任务。
- **安全（式 4–5）：** 惩罚非预期机器人–场景接触与插入任务的卡死力；导出时按接触幅度/持续时间、插入力、容器位移过滤。塑形把通过率从 31% 提到 50%（PnP-Banana）、24% 提到 46%（FMB-SqCircle）。
- **实验设置（§4.1）：** Franka Research 3，Isaac Sim；四任务 PnP-Banana、StackCube、FMB-Round、FMB-SqCircle；每任务 20 条真机遥操作示范（H20）；各生成方法给 3000 条成功示范；下游为 DROID 预训练 π0.5，仿真 3 种子×50 回合，真机 20 次零样本部署。基线：MP、MimicGen、PGDG、PLD、ExpertGen、P-RFCL（α=0）。
- **主结果（§4.2、图 3）：** 四任务平均成功率，仿真 62% 对最强基线 40%，真机 78%（文中 77.5%）对 39%；最大增益仿真 FMB-Round +23 个百分点，真机 PnP-Banana +45。MimicGen/PGDG 成功率较强但轨迹长（继承人类次优动作）；P-RFCL 最短但覆盖窄、偏离后难恢复。
- **分析（§5）：** 用 Vendi 分数度量固定初始场景的多样性与全 3000 条的覆盖，DiscoDemo 两项均最高；数据集级覆盖与下游成功的关联强于固定场景多样性；α 增大时下游成功率先升后降，峰值落在中间值（逐任务 0.1 到 1）。
- **成本（图 5d、7）：** 轻量状态策略 + 大规模并行，端到端 GPU 时远低于反复调用大视觉先验的 PLD/ExpertGen；多样性目标使训练略变长；并行课程比原版 RFCL 在相同设置下快约 17 倍（8.2 小时对 29 分钟，仅 reverse 阶段，墙钟时间）。
- **与纠错数据互补（图 8）：** 叠加 DAgger/DART 后 P-RFCL 与 DiscoDemo 均提升，DiscoDemo 仍更强。
- **局限（原文）：** 依赖精确仿真、特权状态与可重置、明确成功信号；仅四个桌面任务、sim-to-real 差距不大；安全准则任务特定；多样性目标不直接优化下游效用。

## 对 wiki 的映射

- [`wiki/overview/embodied-station-2026-10-29-58-papers-technology-map.md`](../../wiki/overview/embodied-station-2026-10-29-58-papers-technology-map.md)
- [`wiki/entities/paper-discodemo.md`](../../wiki/entities/paper-discodemo.md)
