# ResMimic 论文通读（残差 loco-manipulation 视角）

> 来源归档（原文通读 · 支撑 `wiki/entities/paper-resmimic.md`）

- **标题：** ResMimic: From General Motion Tracking to Humanoid Whole-body Loco-Manipulation via Residual Learning
- **作者：** Siheng Zhao, Yanjie Ze, Yue Wang, C. Karen Liu, Pieter Abbeel, Guanya Shi, Rocky Duan（Amazon FAR / USC / Stanford / Berkeley / CMU）
- **类型：** paper / humanoid / loco-manipulation / residual-learning
- **原文：** <https://arxiv.org/abs/2510.05070>（2026-11 通过 arXiv PDF v2 获取全文）
- **一句话说明：** 先在大规模人类 MoCap 上训练任务无关的 GMT 策略，再逐任务训练轻量残差策略 Δa 叠加其动作，辅以点云物体跟踪奖励、接触奖励与虚拟物体力课程，在 Unitree G1 上实现全身接触的 loco-manipulation。
- **核对范围：** 通读摘要、§I 引言、§II 相关工作、§III 方法（两阶段、GMT 训练、残差策略、奖励与早停）、§IV 实验（Table I、Fig. 5–8、真机评测与两项消融）、§V 结论及参考文献列表。该论文无附录；Table I 的逐项数值按 PDF 抽取核对。未阅读被引论文（TWIST、GMR 等）原文，其细节仅依赖 wiki 既有页面。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题与洞察（§I）：** GMT 能复现多样人类动作但不感知物体；现有 loco-manipulation 依赖分阶段控制器 [10] 或手工数据管线 [11]。平衡、迈步、伸手等跨任务共享，只有细粒度物体交互需要适配，故用残差。
- **MDP（§III）：** 状态含本体 s^r、物体 s^o、动作目标 ŝ^r、物体目标 ŝ^o；动作为目标关节角，经 PD 执行；两阶段均用 PPO。
- **两阶段（§III-A）：** a^gmt = π_GMT(s^r, ŝ^r)；π_Res(s^r, s^o, ŝ^r, ŝ^o) = Δa；a = a^gmt + Δa；残差策略逐任务训练，优化 r^m + r^o。
- **GMT（§III-B）：** AMASS + OMOMO，>15,000 条、约 42 小时，滤除爬楼梯等动作，经 GMR 重定向；单阶段 RL、无特权信息；本体观测含历史窗 t−10:t，参考含 t−10:t+10；奖励与域随机化沿用 TWIST。
- **残差（§III-C）：** 参考由 OptiTrack 同时采集人与物；Δa ∈ R^29；末层 Xavier uniform 小增益初始化使初始输出接近 0；物体状态含位置、朝向、线/角速度。
- **虚拟物体力课程：** PD 虚拟力 F = kp(p̂ − p) − kd v 与力矩 T，增益逐步衰减；动机是重定向穿透导致推倒物体、策略退缩，以及大质量物体不稳定。
- **物体跟踪奖励：** 从 mesh 采样 N 点，r^o = exp(−λ_o Σ‖P_i − P̂_i‖²)，同时覆盖平移与旋转，免去位姿项权重调参。
- **接触奖励与早停：** 参考接触 ĉ = 1(‖d̂‖ < σ_c)，链节为躯干/髋/臂（不含脚）；r^c = Σ ĉ·exp(−λ/f)；早停增加两条件：物体点云偏差超过 σ_o，或必需接触丢失超过 10 连续帧。
- **Table I（MuJoCo sim-to-sim，四任务均值）：** SR——Base 10%、从头训练 0%、微调 7.5%、ResMimic 92.5%（Kneel 90 / Carry 100 / Squat 80 / Chair 100）；收敛迭代——从头 4500、微调 2400、ResMimic 1300；物体误差 E_o 均值 ResMimic 0.12，Base 0.61；关节误差 E_j 上 Base 最低（0.89 对 1.29）。
- **结论文字（Q1–Q3）：** 从头训练在 IsaacGym 有部分成功，迁移到 MuJoCo 全崩；微调无法加入物体观测，且会覆盖 GMT 的泛化；残差更易推广。
- **真机（§IV-C）：** G1（29 DoF，1.3 m）。图 1 的结果为盲部署（无物体状态输入）；非盲部署用 MoCap 物体状态，演示 11 次随机初姿抬箱、连续跪姿抬箱、外力扰动反应。4.5 kg 箱子（G1 手腕载荷约 2.5 kg），椅子 4.5 与 5.5 kg。
- **消融（§IV-D）：** 无虚拟力时，手臂穿透导致物体被撞倒、策略陷入退缩的局部最优；无接触奖励时策略只用手腕（IsaacGym 可成功但迁移失败），有接触奖励则用躯干加手臂。
- **结论（§V）：** 论文未设专门的局限性章节；"每任务需残差训练"与"依赖 MoCap 物体状态"属 wiki 页面基于方法描述的归纳。

## 对 wiki 的映射

- [`wiki/entities/paper-resmimic.md`](../../wiki/entities/paper-resmimic.md)
