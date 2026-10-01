# TactileStep: Sole Tactile Learning for Regulating Foot-Terrain Interaction in Humanoid Locomotion（arXiv:2609.28959）

> 来源归档（PDF + 项目页 + 公众号索引）

- **标题：** TactileStep: Sole Tactile Learning for Regulating Foot-Terrain Interaction in Humanoid Locomotion
- **作者：** Zizhuo Wang *、Ming-Ju Lee *、Shaoting Zhu、Haozhe Lou、Hang Zhao †、Yiming Li †（* 共一；† 通讯）
- **机构：** 清华大学（Tsinghua University）
- **venue：** CoRL 2026（项目页标注 **Spotlight**）
- **arXiv：** <https://arxiv.org/abs/2609.28959>
- **PDF：** <https://arxiv.org/pdf/2609.28959>
- **项目页：** <https://tactilestep.github.io/> → [站点归档](../sites/tactilestep-github-io.md)
- **代码：** （截至 2026-10-01 未列）
- **开源状态：** 待发布
- **类型：** paper / humanoid / locomotion / tactile / reinforcement-learning / perceptive-parkour
- **平台：** Unitree G1，29 DoF；机载 **薄型压力鞋垫** + 深度 + 本体
- **入库日期：** 2026-09-27（公众号 stub）；**2026-10-01** 深度 ingest（PDF + 项目页）

## 为什么值得保留

- 把 **足底阵列触觉** 做成 **部署期 actor 观测**（非仅 critic 特权），直接闭合 **触地冲击与支撑质量**，补感知跑酷「能过但踩得重」的短板。
- **特征级 sim2real**：仿真与硬件共享 $\bar F$、CoP、$\bar A$，避免 raw taxel 高维迁移；与 [QuietWalk](../../wiki/entities/paper-quietwalk-humanoid-locomotion.md)（训练期 GRF 惩罚、部署不用 sole 反馈）形成对照轴。
- 强 **Hiking in the Wild** 基线 + 系统消融，便于和 [人形运控奖励](../concepts/humanoid-policy-reward-functions.md)、[触觉感知](../concepts/tactile-sensing.md) 交叉。

## 核心摘录（≥3 条）

1. **观测与优化：** POMDP + **PPO**；每只脚触觉 $\mathbf{x}^f_t=[\bar F^f_t, \mathbf{p}^{cop,f}_t, \bar A^f_t]$；actor 为 **本体历史** $\oplus$ **触觉历史** $\oplus$ **深度历史** $\mathcal{H}_t$（深度同 [46] Hiking）；**双 critic** 共享特权观测（含足端竖直速度、四相位 one-hot），分别估计 **稠密/稀疏** 奖励组回报。
   - **对 wiki 的映射：** [paper-tactilestep](../../wiki/entities/paper-tactilestep.md)「核心原理 / 观测空间」；[humanoid-policy-observation-inputs](../../wiki/concepts/humanoid-policy-observation-inputs.md)。

2. **触觉仿真（Isaac Sim）：** 每足 **M=60** taxels；raycast 得 gap 与地形法向 → 对齐加权分配刚体法向接触力 → **kNN 空间扩散** 抑制单点接触 → 提取 $F_{tac}$、$A_{tac}$、$\mathbf{p}_{cop}$ 并归一化；硬件侧 MLP 将原始读数标定到力。
   - **对 wiki 的映射：** 实体页「触觉仿真」与 Mermaid 流程；[sim2real](../../wiki/concepts/sim2real.md) 特征对齐叙事。

3. **四相位 + 相位条件奖励：** 在线推断 $s^f\in\{\text{Swing, PreLanding, Landing, Stance}\}$；PreLanding 惩罚向下速度/加速度；Landing 惩罚 $\bar F$ 与 $\Delta\bar F$ 及窗口峰值；Stance 奖励 $\bar A$、CoP margin、惩罚 CoP 跳变。
   - **对 wiki 的映射：** [humanoid-policy-reward-functions](../../wiki/concepts/humanoid-policy-reward-functions.md)「步态与接触」类奖励的 **触觉量化** 实例。

4. **实验协议：** Isaac Lab，2048 并行 env，RTX 4090；仿真每策略每地形 **4096** trials；真机每条件 **20** 样本；指标：$F_{impact}$、$A_c$、峰值 A 加权噪声 $L_{A,peak}$。
   - **对 wiki 的映射：** 实体页「评测」表；[stair-obstacle-perceptive-locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md) 策展。

5. **主要结论（摘要 + 真机表）：** 相对 Hiking 基线，触地峰值力最多 **−48.8%**（平台上升），峰值噪声最多 **−30.1 dB**（楼梯下降），支撑接触面积最多 **+23.8%**（楼梯下降）；成功率持平或略高，**能耗与速度 RMSE 略升** 为显式代价。
   - **对 wiki 的映射：** 实体页「结论」；与 [Hiking in the Wild](../../wiki/entities/paper-hiking-in-the-wild.md) 对比表。

## 对 wiki 的映射

- 主实体：[paper-tactilestep.md](../../wiki/entities/paper-tactilestep.md)
- 索引来源：[公众号 12 篇](../blogs/wechat_embodied_station_12_papers_recover_wam_2026-09-27.md)、[深兰科周报](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
