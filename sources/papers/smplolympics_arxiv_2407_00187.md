# SMPLOlympics: Sports Environments for Physically Simulated Humanoids

> 来源归档（ingest）

- **标题：** SMPLOlympics: Sports Environments for Physically Simulated Humanoids
- **缩写：** **SMPLOlympics**
- **类型：** paper / simulation benchmark / physics-based humanoid sports
- **arXiv：** <https://arxiv.org/abs/2407.00187>（PDF：<https://arxiv.org/pdf/2407.00187>）
- **项目页：** <https://smplolympics.github.io/SMPLOlympics>
- **代码：** <https://github.com/SMPLOlympics/SMPLOlympics>
- **作者：** Zhengyi Luo, Jiashun Wang, Kangni Liu, Haotian Zhang, Chen Tessler, Jingbo Wang, Ye Yuan, Jinkun Cao, Zihui Lin, Fengyi Wang, Jessica Hodgins, Kris Kitani（CMU + NVIDIA）
- **入库日期：** 2026-06-12
- **一句话说明：** 在 **SMPL / SMPL-X 兼容** 的统一仿真人形上提供 **10 项奥运风格运动**（田赛 + 球类对抗）环境与基线 reward/state 设计，用 **PPO / AMP / PULSE** 对照证明「强运动先验 + 简单任务奖励」可学到类人策略，并给出 **TRAM→PHC** 视频示范管线。

## 核心摘录

### 1) 环境与统一人形

| 类别 | 运动 |
|------|------|
| **个人项目** | 高尔夫、标枪、跳高、跳远、跨栏 |
| **对抗 / 团队** | 乒乓球、网球、击剑、拳击、足球（点球/1v1/2v2）、篮球（罚球） |

- **SMPL 人形**（24 关节，69 维动作）：无精细手指任务。
- **SMPL-X 人形**（52 关节，153 维动作）：标枪、篮球等需手指任务。
- **仿真**：Isaac Gym；策略 30 Hz，仿真 60 Hz；单卡 RTX 3090 约 1–3 天/任务。

### 2) Goal-conditioned RL  formulation

- 状态 = 本体 \((q_t, \dot{q}_t)\) + 任务目标 \(s^g_t\)（球位、障碍、对手根位姿等，heading 归一化）。
- 每项运动提供**可复现的基线 reward**（论文 §4 与附录）；竞争性运动含交替自博弈（一策略冻结、另一策略训练）。

### 3) 视频 → 示范数据管线

1. **TRAM** 从互联网体育转播估计 SMPL 全局轨迹与姿态；
2. **PHC** 物理跟踪修正，使示范更适合 AMP 正样本；
3. 用于高尔夫、网球、乒乓球、足球点球等「有视频 示范」任务。

### 4) 基线结论（摘要）

- **纯 PPO**：任务可完成但动作常不自然（如跳远「弹跳」过栏）。
- **AMP**：易 **reward 冲突** / mode collapse（站定刷判别器奖励、乒乓球只追求回合数忽略落点）。
- **PULSE**（AMASS 预训练 32 维 motion latent）：跨栏 **17.76 s** 完赛、跳高可自发 **Fosbury flop**；缺专项数据时长距离受限。
- **PULSE + AMP**：乒乓球 **Avg Hits 1.83** 显著优于单用 PULSE；课程学习对跳高 1.5 m / 跨栏至关重要。

## 对 wiki 的映射

- 新建实体页：[`wiki/entities/smplolympics.md`](../../wiki/entities/smplolympics.md)
- 交叉更新：
  - [`wiki/entities/zhengyi-luo.md`](../../wiki/entities/zhengyi-luo.md) — 作者与 PULSE/PHC 脉络
  - [`wiki/methods/amp-reward.md`](../../wiki/methods/amp-reward.md) — 体育场景 AMP 与任务奖励冲突
  - [`wiki/methods/table-tennis-strategy-skill-learning.md`](../../wiki/methods/table-tennis-strategy-skill-learning.md) — 同作者 Jiashun Wang 的专项乒乓球分层控制（SIGGRAPH 2024）
  - [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md) — 仿真人形运动 benchmark 挂接

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2407.00187>
- 项目页：<https://smplolympics.github.io/SMPLOlympics>
- 代码：<https://github.com/SMPLOlympics/SMPLOlympics>
