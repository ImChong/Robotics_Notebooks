# Barkour：四足动物级敏捷评测基准（arXiv:2305.14654）

> 论文来源归档（ingest）

- **标题：** Barkour: Benchmarking Animal-level Agility with Quadruped Robots
- **类型：** paper / quadruped / benchmark / reinforcement-learning / sim2real
- **arXiv：** <https://arxiv.org/abs/2305.14654> · PDF：<https://arxiv.org/pdf/2305.14654.pdf>
- **机构：** Google DeepMind（Ken Caluwaerts, Atil Iscen, J. Chase Kew, Wenhao Yu, Tingnan Zhang 等）
- **补充材料入口：** <https://sites.google.com/view/barkour>
- **入库日期：** 2026-05-18
- **一句话说明：** 提出 **5m×5m** 障碍课（起终点台、绕杆、A 字坡、宽跳）与 **时间型敏捷分** \(R_{\text{agility}}\)，并给出 **三专长 PPO（LeggedGym/Isaac Gym）+ 高层导航状态机** 与 **专长蒸馏的 Locomotion-Transformer 通才策略** 两套基线，在 **自研四足** 上 **零样本 sim2real** 完成课程（论文称约为犬用时的约一半速度量级作为对照叙事）。

## 核心摘录（面向 wiki 编译）

### 1) 基准与计分

- **要点：** 借鉴犬敏捷赛规则简化：满分 1.0 表示在 **规定总时长** \(t_{\text{allotted}}\) 内无失误完成；超时按秒扣 0.01，跳过/失败障碍各扣 0.1；目标平均速度取 **1.69 m/s**（可按更大机器人缩放）。全课 **名义长度约 18 m**、**规定时间约 10.64 s**（Table I）。
- **对 wiki 的映射：** [`wiki/entities/paper-barkour-quadruped-agility-benchmark.md`](../../wiki/entities/paper-barkour-quadruped-agility-benchmark.md)

### 2) 专长策略与观测

- **要点：** 三类专长：**全向 uneven 行走（OWP）**、**爬坡 SCP（5°→33° 课程）**、**宽跳 JP（平地→间隙→强随机化三阶段）**；观测含 **速度指令**、**本体**（投影重力、关节角、偏航角速度等）、**局部高度场**（随任务定制网格）、**0.3 s 历史**；动作空间为 **相对站立名义姿态的电机角目标**（PD 跟踪接口语境）。
- **对 wiki 的映射：** 同上实体页；与 [RL+PD 动作接口](../../sources/papers/rl_pd_action_interface_locomotion.md) 文献族对照阅读。

### 3) 域随机化与敏捷 sim2real

- **要点：** 在 Rudin 等默认 DR 之外，对 **>1 m/s** 敏捷动作补充 **躯干惯量、电机建模、关节静摩擦** 等随机化（Table II），报告对 **跳跃/大坡** 迁移关键。
- **对 wiki 的映射：** [`wiki/concepts/domain-randomization.md`](../../wiki/concepts/domain-randomization.md)、[`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)

### 4) Locomotion-Transformer（通才）

- **要点：** 用三专长在 **随机台阶/楼梯/间隙/坡** 等环境 rollout 收集 **17636 episodes（论文写作约 57.58 机器小时）**；因果 Transformer 在固定上下文 **W=15（0.3 s）** 下拼接 **本体序列 + 动作历史 + 融合高度图 token**，**L2 回归** 预测下一步动作。
- **对 wiki 的映射：** 同上实体页；[`wiki/methods/reinforcement-learning.md`](../../wiki/methods/reinforcement-learning.md)

### 5) 高层导航

- **要点：** 状态机 + **路点**（位置/朝向/容差）；专长路线下每点还带 **行为类型** 以切换策略；通才仅需速度指令跟踪。
- **对 wiki 的映射：** 同上实体页

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
- [x] 与开源 `barkour_robot` / Menagerie 资产版本（v0 vs vB）在实体页脚注对齐
