# Learning Roller-Skating Motions of Humanoid Robots Based on Adversarial Motion Priors（arXiv:2607.10815）

> 来源归档（ingest）

- **标题：** Learning Roller-Skating Motions of Humanoid Robots Based on Adversarial Motion Priors
- **类型：** paper / humanoid locomotion / passive-wheel skating / AMP / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2607.10815>
- **arXiv HTML：** <https://arxiv.org/html/2607.10815v1>
- **PDF：** <https://arxiv.org/pdf/2607.10815>
- **项目页：** <https://cyk579.github.io/Roller-Skating/>
- **机构：** 清华大学（Tsinghua University）
- **作者：** Yunkang Cheng†、Yutong Wu†、Menghan Li、Shihe Zhou、Mingguo Zhao*（†共同一作，*通讯作者）
- **硬件：** Booster T1 人形 + 被动轮滑改装（23 主动关节 + 每脚 4 个被动轮共 8 被动 DoF）
- **仿真：** NVIDIA Isaac Lab；Push Glide 评测另用 MuJoCo
- **发表日期：** 2026-07-14
- **入库日期：** 2026-07-20
- **一句话说明：** 被动轮滑人形上，用 **GMR 重定向人体轮滑 MoCap** 为 **Pump Glide / Push Glide** 两套独立 AMP 参考集，配合 **切片圆柱被动轮碰撞模型** 与 **AMP-PPO** 分别学两种推进机制，并在仿真与 T1 真机验证速度跟踪、转向与长程滑行。

## 摘要级要点

- **任务难点：** 被动轮无轮端驱动，推进靠腿部摆动、重心转移与轮地摩擦；支撑点随滚动连续移动，侧向摩擦低、姿态速度耦合强，比常规模型预测控制与纯任务奖励 RL 更难。
- **仿真核心 — 切片圆柱轮：** 对比 STL mesh、sphere-band、整球与 1/3/9/15 片圆柱；整球在轮宽 $w=23.13$ mm、半径 $R=32$ mm 时约 **48.2%** 体积落在真实轮宽外，roll 角 $>21.2°$ 会产生虚假侧向支撑；**9 片圆柱** 在滚动稳定性、几何保真与训练吞吐（约 $7.13\times10^3$ steps/s）间折中。
- **两种步态：** **Pump Glide** — 周期对称开合「沙漏」轨迹，协调脚间距与板向；**Push Glide** — 单腿蹬地、另一腿滑行支撑交替，需支撑切换与重心转移。
- **AMP-PPO 管线：** 人体 MoCap → GMR 重定向 → 平滑/重采样/滤除穿地自碰 → 短状态转移样本；两 gait **独立判别器、策略与任务奖励**；策略仅输出 21 个非轮关节目标，50 Hz PD，轮关节被动。
- **奖励分工：** Pump 主要靠 AMP 风格 + 速度/姿态/脚距；Push 额外用轮腾空比、轮速、支撑腿切换、单支撑时长等 **课程化** 项抑制早期跳跃策略。
- **真机结果：** Pump 在 $0.10$–$0.45$ m/s 完成率 0.766–0.813，100 s 剖面总行程 39.5 m；Push 速度单调随命令上升但存在 **增益偏高**（0.1→0.5 m/s 命令对应实际约 0.366→1.594 m/s），可能与 Isaac/MuJoCo 轮模型差异有关。

## 核心摘录（面向 wiki 编译）

### 与 SKATER 等相邻工作

| 维度 | 本文（Tsinghua AMP 轮滑） | [SKATER](humanoid_pnb_skater-synthesized-kinematics-for-advanced-trave.md)（arXiv:2601.04948） |
|------|---------------------------|-----------------------------------------------------------------------------------------------|
| 推进机制 | **Pump Glide + Push Glide** 两套独立策略 | Pump Glide（swizzle）为主 |
| 风格先验 | **AMP 对抗运动先验**（人体 MoCap） | 任务奖励 + 课程（论文路线不同） |
| 轮模型 | **9 片圆柱** 被动轮碰撞近似 | 未在本归档展开 |
| 平台 | Booster T1 + 被动轮滑 | 人形轮滑（待深读） |
| 开源 | 截至入库日 **项目页无 GitHub** | 待深读 |

### AMP 训练要点

- 判别器：5 帧拼接 AMP 状态 $z_t^m\in\mathbb{R}^{300}$，Wasserstein 形式 + GP；$r_{amp}=c_{amp}(1+\tanh(0.4D(z)))$，$c_{amp}=2.0$
- 总奖励混合：Pump $0.40\,r_{amp}+0.60\,r_{task}$；Push $0.45\,r_{amp}+0.55\,r_{task}$
- 80% 环境从参考 motion 随机帧初始化；域随机化含观测噪声、摩擦、质量、CoM、PD 增益、轮阻尼

## 对 wiki 的映射

- 沉淀实体页：[被动轮人形轮滑 AMP（arXiv:2607.10815）](../../wiki/entities/paper-roller-skating-amp-humanoid-passive-wheels.md)
- 交叉补强：[Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[SKATER 轮滑索引](../../wiki/entities/paper-notebook-skater-synthesized-kinematics-for-advanced-trave.md)

## 当前提炼状态

- [x] 摘要、轮模型、双 gait AMP 管线、真机指标摘录
- [x] wiki 实体页与 locomotion 交叉链接规划
- [ ] 待作者公开代码后补 `sources/repos/`
