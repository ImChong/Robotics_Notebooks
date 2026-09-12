---
type: method
tags: [rl, imitation-learning, locomotion, humanoid, sampling, diffusion, paper, motion-control, body-system-stack, bfm, behavior-foundation-model, stanford, berkeley]
status: complete
updated: 2026-09-12
code: https://github.com/HybridRobotics/whole_body_tracking
venue: "2026 · Science Robotics"
arxiv: "2508.08241"
doi: "10.1126/scirobotics.adx8924"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-04-wbt-base.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../overview/bfm-category-05-hierarchical-control.md
  - ../methods/beyondmimic.md
  - ./imitation-learning.md
  - ./deepmimic.md
  - ./egm-efficient-general-mimic.md
  - ./sonic-motion-tracking.md
  - ../concepts/armature-modeling.md
  - ../concepts/curriculum-learning.md
  - ../concepts/reward-design.md
  - ../entities/paper-extreme-rgmt.md
  - ../entities/paper-agile-humanoid-loco-manipulation.md
  - ../entities/paper-pfm-hr.md
  - ../entities/paper-umr-unified-motion-retargeting.md
  - ../entities/paper-vibe.md
sources:
  - ../../sources/repos/beyondmimic-reproduction.md
  - ../../sources/papers/motion_control_projects.md
  - ../../sources/papers/bfm_awesome_beyondmimic_arxiv_2508_08241.md
  - ../../sources/papers/loco_manip_161_survey_004_beyondmimic.md
  - ../../sources/papers/humanoid_rl_stack_15_beyondmimic_from_motion_tracking_to_versatile_hu.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/blogs/wechat_shenlan_beyondmimic_science_robotics_2026-09-10.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
  - ../../sources/papers/agile_arxiv_2603_20147.md
  - ../../sources/papers/pfm_hr_arxiv_2608_03227.md
summary: "BeyondMimic（*Science Robotics* 2026，DOI adx8924）是两阶段人形全身控制框架：阶段 ① 用锚点相对跟踪 + 极简奖励 + 失败率采样在共享超参下批量学高动态技能；阶段 ② 把跟踪教师蒸馏进潜空间状态–动作扩散模型，测试时用 classifier guidance 零样本完成航点、摇杆、关键帧补全与避障。"
---

# BeyondMimic

**BeyondMimic** 是由 Hybrid Robotics 等团队开发的高性能机器人动作模仿框架。相比早期的 DeepMimic 或 AMP，BeyondMimic 更侧重于从仿真到真实物理世界的无缝迁移，并在 **Isaac Lab** (IsaacLab) 环境中得到了广泛验证。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BeyondMimic | BeyondMimic Framework | 高精度仿真人形动作模仿框架 |
| IL | Imitation Learning | 参考轨迹跟踪式模仿学习 |
| RL | Reinforcement Learning | 仿真中 PPO 等优化跟踪策略 |
| Isaac Lab | NVIDIA Isaac Lab | 主要验证与训练环境 |
| Sim2Real | Simulation to Real | 强调物理建模与采样以促迁移 |
| CG | Classifier Guidance | 测试时用代价函数梯度引导扩散采样朝新目标优化 |
| SciRob | Science Robotics | 正式发表渠道（2026-08-26，DOI adx8924） |
| SDF | Signed Distance Field | 避障等任务中构造障碍物排斥代价 |

## Survey 坐标（策展索引）

### 在 42 篇 RL 运动控制身体系统栈中

| 字段 | 内容 |
|------|------|
| 编号 | 15/42 |
| 系统栈层 | 02 参考跟踪 · 通用控制 |
| 索引来源 | [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) |

### 在 BFM 41 篇技术地图中

| 字段 | 内容 |
|------|------|
| 编号 | 35/41 |
| 分组 | 05 Hierarchical control |
| 索引来源 | [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) |

### 在人形 Loco-Manip 161 篇中

| 字段 | 内容 |
|------|------|
| 槽位 | 004/161 |
| 分组 | 01 运控基座与通用全身跟踪 |
| 分类 hub | [loco-manip-161-category-01-motion-base-wbt](../overview/loco-manip-161-category-01-motion-base-wbt.md) |
| 索引来源 | [具身智能研究室 · 161 篇人形 Loco-Manip 长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) |

## 论文信息（*Science Robotics* · arXiv:2508.08241）

| 字段 | 内容 |
|------|------|
| 完整标题 | *BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion* |
| 作者 | Qiayuan Liao、Takara E. Truong、Xiaoyu Huang、Guy Tevet、Koushil Sreenath、C. Karen Liu |
| 机构 | 加州大学伯克利分校（Hybrid Robotics）；斯坦福大学 |
| 正式发表 | [*Science Robotics*（2026-08-26）](https://doi.org/10.1126/scirobotics.adx8924)，DOI `10.1126/scirobotics.adx8924` |
| arXiv 版本 | v1 2025-08-11 → v4 2025-11-13（[2508.08241](https://arxiv.org/abs/2508.08241)） |
| 代码 | <https://github.com/HybridRobotics/whole_body_tracking>（跟踪阶段开源实现） |
| 项目页 | <https://beyondmimic.github.io/> |

论文叙事分两个阶段：**① 紧凑 motion-tracking 公式**——单一 MDP 设置与共享超参覆盖高动态技能；**② 统一潜空间扩散 + classifier guidance**——把跟踪技能升格为可组合、可引导的通用控制（详见下文「[第二阶段](#第二阶段统一潜空间扩散与测试时引导)」）。本页前半部分的物理建模与采样细节属于阶段 ①。

> **训练–测试解耦（文内总判断）**：训练只学运动先验，不预设下游任务；航点、摇杆、避障、关键帧补全等任务语义在**测试时**以可微代价函数注入扩散采样，而非为每个任务重训网络。

## 端到端数据流（概览）

下面用一张流程图把「参考动作 → 仿真环境 → 策略学习 → 部署」的主干串起来；具体张量布局随实现（官方仓库与 [robot_lab](../../sources/repos/robot_lab.md) 等 fork）略有差异，但信息流一致。

```mermaid
flowchart TD
  subgraph ref["参考与预处理"]
    M[参考动作序列<br/>关节角 / 根位姿 / 速度等]
    R[重定向 / 时间对齐<br/>可选]
  end
  subgraph sim["仿真与奖励"]
    E[Isaac Lab 环境<br/>精确 armature + PD]
    O[观测构造<br/>本体感知 + 参考相对量 + 历史堆叠]
    RW[统一任务空间奖励<br/>位姿 / 速度跟踪项]
    S[失败率统计<br/>按片段更新采样权重]
  end
  subgraph rl["学习"]
    P[PPO<br/>Actor-Critic]
  end
  subgraph dep["输出"]
    Pi[策略 π<br/>部署时可仅保留观测→动作映射]
  end
  M --> R --> E
  E --> O
  E --> RW
  E --> S
  S --> E
  O --> P
  RW --> P
  P -->|动作指令| E
  P --> Pi
```

## 源码运行时序图

官方跟踪阶段实现 [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) 基于 Isaac Lab，参考动作用 WandB Registry 管理：先用 `scripts/csv_to_npz.py` 把重定向动作转成参考 npz 并注册，再用 `scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0` 训练，`scripts/rsl_rl/play.py` 回放与导出。一次完整运行的模块交互如下（具体张量与命令行以仓库 README 为准）：

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant PRE as scripts/csv_to_npz.py
    participant WB as WandB Registry
    participant TR as scripts/rsl_rl/<br/>train.py · play.py
    participant ENV as Isaac Lab 并行环境<br/>Tracking-Flat-G1-v0
    participant PPO as rsl_rl PPO<br/>OnPolicyRunner
    U->>PRE: 输入重定向动作 CSV<br/>（LAFAN1 / GMR 输出等）
    PRE->>PRE: 运动学回放补全<br/>身体位姿与速度
    PRE->>WB: 注册参考动作 .npz
    U->>TR: train.py --task=Tracking-Flat-G1-v0<br/>--registry_name=指定动作
    TR->>WB: 拉取参考动作
    TR->>ENV: 创建并行环境<br/>加载参考动作命令项
    loop 每次 PPO 迭代
        PPO->>ENV: 批量动作（关节目标 → PD）
        ENV->>ENV: 物理步进 + 失败率<br/>自适应片段采样
        ENV-->>PPO: 观测 + 统一任务空间<br/>跟踪奖励
        PPO->>PPO: GAE + PPO 更新
        PPO->>WB: 曲线 / 视频 / checkpoint
    end
    U->>TR: play.py --wandb_path=训练运行号
    TR-->>U: 加载 checkpoint 回放<br/>并导出部署用策略
```

- **训练与部署解耦**：本仓库只覆盖「参考动作 → 跟踪策略」的训练闭环；真机部署控制器在配套 deploy 仓库，与上文「端到端数据流」中的 `策略 π` 输出衔接。
- **动作即资产**：参考动作与 checkpoint 全走 WandB Registry，换动作只改 `--registry_name`，与「失败率驱动自适应采样」一起构成多技能批量训练的工程底座。

## 输入与输出：和实现对齐时看什么

本节按「环境 / 策略」两侧说明，便于你对照代码里的 `observation`、`action`、`reward` 配置与 TensorBoard 曲线。

### 1. 参考侧输入（教师信号，非策略网络输入）

| 类型 | 常见内容 | 在训练中的作用 |
|------|-----------|----------------|
| 参考轨迹 | 逐帧根位姿、关节角、根线速度 / 角速度等 | 定义「要像谁」；与 [DeepMimic](./deepmimic.md) 族一样属于 **轨迹跟踪** 范式 |
| 时间索引 | 当前应对齐到参考的第几帧 / 相位 | 决定奖励里与哪一段参考比较；长序列上常配合 **失败驱动采样** 决定 reset 片段 |
| 坐标变换 | 根坐标系、质心 / 骨盆局部系 | 任务空间奖励在 **统一坐标系** 下算误差，避免关节空间手工拼凑 |

参考数据通常 **不** 作为原始像素输入进策略；策略看到的是已在观测里编码好的 **相对几何与速度误差**（见下）。

### 2. 策略观测（Policy 输入 · 阶段 ① 跟踪）

BeyondMimic 的 **motion-tracking MDP 刻意不做历史堆叠**：论文 III-B 写明跟踪框架 **does not have a history**，观测为**单步向量**，以降低过拟合仿真特有时序、提升 sim2real（与 [Science Robotics 中文导读](../../sources/blogs/wechat_shenlan_beyondmimic_science_robotics_2026-09-10.md) 及消融一致——**不恰当增大观测历史长度反而损害迁移**）。常见组块包括：

| 组块 | 含义 | 调参 / 排错提示 |
|------|------|------------------|
| 参考相位 | 参考关节角 / 角速度 $\mathbf{c}$，仅作相位信息，**非**直接关节跟踪目标 | 与锚点相对跟踪配合；勿与奖励坐标混用 |
| 锚体位姿误差 | 参考体（通常为 root / torso）相对当前的三维位置误差 + 旋转矩阵前两列（Rot6D 风格） | 消融显示 Rot6D 连续旋转表示优于四元数 / 轴角 |
| 本体状态 | 根 twist（根坐标系）、关节角 / 角速度、上一步动作 | 无可靠状态估计时可省略线性位置项与线性 root twist |
| ~~历史堆叠~~ | **阶段 ① 不使用** | 社区 fork 若自行加 history，需单独评估 sim2real |

**锚点相对跟踪**：以参考体 $b_{\text{ref}}$ 为锚，将各连杆目标位姿表达为相对锚点的变换 $\hat{T}_b$，允许全局 xy / yaw 合理漂移而保留动作风格——这是 compact MDP 能在扰动与 sim2real gap 下仍保持自然度的关键设计之一。

### 3. 策略动作（Policy 输出）

在 Isaac Lab 类人形任务里，动作多为 **目标关节位置 / 速度** 或 **在 PD 之上的残差**，由底层 PD + 精确 armature 模型执行。部署时输出的是 **控制指令**（与训练时相同的接口），而不是奖励或参考索引。

| 输出 | 典型语义 | 备注 |
|------|-----------|------|
| 动作向量 | 各关节目标或残差，维度 = 可控自由度 | 与 URDF / 执行器模型一致；armature 与增益错误会表现为 **同样策略在实物上发散** |
| 隐变量 | 一般无 | 若使用 VAE 等才会多出头；标准 BeyondMimic 叙述以 PPO 为主 |

### 4. 环境反馈与奖励分解（理解曲线用的「物理含义」）

统一任务空间奖励通常可看成若干项的加权和（具体权重看配置）：

- **位置 / 姿态误差**：各关键连杆与参考的平移、旋转差；决定「像不像」。
- **线速度 / 角速度匹配**：决定「节奏与动态是否一致」，避免「pose 对了但发软或发飘」。
- **正则项**（若实现中有）：能量、关节限位、脚滑惩罚等；防止为降位置误差而 **利用仿真漏洞**。

失败率驱动的采样改变的是 **哪些状态被反复见到**，而不是直接改变奖励公式；因此在曲线上更多体现为 **有效 episode 长度、终止原因分布** 的变化，而非单条 reward 斜率突变。

## 训练曲线：每条在说什么、怎样算「好」

以下按「PPO 通用 → 模仿任务特有关」顺序；指标名以 RSL-RL / TensorBoard 常见命名为例。

### PPO 与价值函数

| 曲线 / 指标 | 健康时大致长什么样 | 常见异常与含义 |
|-------------|-------------------|------------------|
| `episode_reward` / return | 前期快速上升后进入平台期；平台期仍有小幅抖动正常 | **长期持平或下滑**：参考进度跟不上、终止过难、或奖励坐标与观测不一致 |
| `policy_loss` | 小幅波动，无单向爆炸 | **持续飙高**：步长过大、优势估计方差大、或 reward scale 突变 |
| `value_loss` | 随训练缓慢下降 | **先降后升且伴随 return 崩**：critic 过拟合或环境非平稳（如突然改奖励权重） |
| `entropy` | 逐渐缓慢下降；保留一定宽度 | **极快掉到接近 0**：探索不足，易卡在局部跟踪模态；**长期过高**：可能没学到确定性跟踪 |
| `approx_kl` | 维持在你设定的小阈值附近（如 0.01–0.03 量级，依实现而定） | **频繁尖峰**：更新过激进；**始终接近 0**：可能学习率过小或梯度被 clip 死 |

判读技巧：**不要单看一条线**。若 return 上升但 `entropy` 骤降且实机变差，多半是策略过拟合仿真可 exploit 的动力学细节（例如不真实 foot friction），应回到 armature / 接触与奖励权重。

### 模仿与跟踪任务特有关

| 曲线 / 指标 | 含义 | 好 / 坏的工程判据 |
|-------------|------|-------------------|
| 分项 reward（若日志拆开） | 位置项 vs 速度项的贡献 | **位置项独高、速度项低**：动作「卡帧」、动态不对；宜检查速度权重或参考微分是否平滑 |
| Episode length | 每回合持续步数 | **逐渐变长** 通常说明更少提前 fall / timeout；配合失败采样时，早期变短有时表示 **正在专攻难点片段**（需结合终止统计看） |
| Success / fall 率（若有） | 是否站住、是否跟完片段 | 比 return 更直观；**成功率 plateau 在低位** 时优先查物理参数而非网络宽度 |
| 脚滑、穿透相关 proxy（若记录） | 接触是否可信 | **单调变差** 说明策略在利用接触模型漏洞；BeyondMimic 路线应先核对 **PD + armature** 再加大域随机 |

### 实操 checklist（看板 5 分钟版）

1. **先看 video / rollout**：return 骗人时，肉眼比任何标量都快。
2. **对齐时间轴**：改奖励权重或参考数据后，旧 run 与新 run 不要横比绝对 return。
3. **看终止原因占比**：timeout 多 = 难或采样太狠；early termination 多 = 平衡或跟踪失败。
4. **对照 sim2real**：若 sim 曲线完美而硬件上发散，优先打开 [Armature](../concepts/armature-modeling.md) 与执行器文档，而不是先加网络层数。

## 核心设计理念

BeyondMimic 提出一个核心观点：**精确的物理建模可以替代大量盲目的域随机化 (Domain Randomization)**。通过缩小仿真与现实在确定性物理量上的差距，策略能更有效地学习到稳健的运动模式。

## 关键技术点

### 1. 精确的物理建模 (Accurate Physical Modeling)
BeyondMimic 强调必须对机器人执行器的反射惯量（[Armature](../concepts/armature-modeling.md)）进行精确计算，并据此设计 PD 增益。

- **Armature 计算**：$I_{arm} = J_{rotor} \cdot G^2$。
- **PD 增益设计**：基于反射惯量计算临界阻尼增益，确保在轻载工况下不振荡，重载下保持柔顺。

### 2. 失败率驱动的自适应采样 (Failure-driven Adaptive Sampling)
在训练长序列动作（如长距离行走或跳舞）时，随机从序列中任意位置 reset 往往效率低下。BeyondMimic 引入了自适应采样：
- **实时评估**：记录每个动作片段（Segment）的训练失败率。
- **权重分配**：失败率越高、难度越大的片段，被采样作为起始位置的概率越大。
- **前瞻卷积**：采样权重考虑当前片段及其后续片段的累计难度，防止机器人卡在“断点”处。

### 3. 统一的任务空间奖励 (Unified Task-space Rewards)
BeyondMimic 并不针对特定关节设计复杂的 reward，而是采用统一的任务空间跟踪项：
- 身体各部位的位置误差与朝向误差。
- 线速度与角速度匹配。
- 支持对特定关键身体部位（如 Pelvis）进行加权优化。

## 主要技术路线

| 模块 | 核心方案 | 目的 |
|------|---------|------|
| **物理建模** | 精确 armature + 关联 PD 增益 | 缩小动力学 Gap，提升部署稳定性 |
| **采样策略** | 失败率驱动的自适应重采样 | 提高对困难动作片段的训练效率 |
| **观测空间** | **单步**本体 + 锚点误差 + 参考相位（**无**历史堆叠） | 阶段 ② 扩散策略另用 $N$ 步 state–action 历史（论文约 $N{=}4$） |
| **奖励函数** | 统一的任务空间跟踪项 + 关节限位 / 平滑 / 自碰撞正则 | 简化奖励设计，保持动作自然度 |

## 训练机制：大道至简

BeyondMimic 阶段 ① 证明在 **共享 MDP + 共享超参** 下，简单 PPO 即可学到极强的高动态模仿，关键在问题 formulation 而非复杂网络：

1. **锚点相对跟踪 + Rot6D 位姿误差**（允许合理全局漂移）。
2. **单步观测、无历史堆叠**（避免过拟合仿真时序；扩散阶段才引入历史窗口）。
3. **精确的 Armature 补偿 + 适度（非暴力）域随机**。
4. **失败率驱动的自适应片段采样**。

论文口径下这套「紧凑公式」的验证方式：约 **2.5 h** 多样化人类动捕；在 **LAFAN1** 的 **14 段约 3 分钟长序列** 上逐条训练跟踪策略，**全部使用同一 MDP 设置与共享超参**（不做逐动作调参），**21** 个代表性片段 **零样本**部署 **Unitree G1**，覆盖侧空翻、旋踢、翻转踢、冲刺跑、舞蹈、倒地起立等高动态技能。

## 第二阶段：统一潜空间扩散与测试时引导

论文题目的后半句（*Versatile Humanoid Control via **Guided Diffusion***）对应第二阶段：跟踪只能「复现已有动作」，而下游任务往往在训练中从未出现。BeyondMimic 的做法是把技能压进一个生成模型，再在测试时"掰"向新目标：

- **技能蒸馏进统一潜空间扩散模型**：把第一阶段多条跟踪专家策略的 rollout 经 **条件 VAE + DAgger** 压入平滑潜空间，再训练 **Transformer 去噪器**；**联合建模未来状态序列与动作序列**（Diffuse-CLoC 式 state–action co-diffusion），而非仅输出关节角。
- **Classifier guidance 做测试时优化**：给定简单的代价函数（如到路点的距离、速度指令、与障碍的 SDF 距离、稀疏关键帧硬约束），在扩散去噪迭代过程中，把代价梯度加到潜变量更新步骤，**无需针对下游任务再训练**；多种代价可**直接相加**（如「一边导航一边避障」）。
- **零样本下游任务**：论文与 *Science Robotics* 实机验证了 **motion inpainting（动作补全）**、**joystick 遥操**、**waypoint 导航** 与 **障碍规避**；动捕 / 外部感知用于路点、障碍几何与部分状态估计，**非**端到端视觉避障。

```mermaid
flowchart LR
  subgraph stage1["阶段 ①：motion tracking"]
    T1[LAFAN1 14 段长序列<br/>单一 MDP + 共享超参]
    T2[逐条跟踪专家策略<br/>PPO + 失败驱动采样]
  end
  subgraph stage2["阶段 ②：guided diffusion"]
    D1[蒸馏进统一<br/>潜空间扩散策略]
    D2[classifier guidance<br/>代价函数梯度引导采样]
  end
  subgraph tasks["零样本下游任务"]
    K1[motion inpainting]
    K2[joystick 遥操]
    K3[waypoint 导航]
    K4[障碍规避]
  end
  T1 --> T2 --> D1 --> D2
  D2 --> K1
  D2 --> K2
  D2 --> K3
  D2 --> K4
```

这一步使 BeyondMimic 与「每任务重训一条 goal-conditioned policy」或「上层规划 + 下层跟踪解耦」的路线区分开：**任务语义在测试时以代价函数注入**，策略本体保持不变；规划与控制在同一扩散模型内 **滚动闭环** 下发动作，规避规划器–控制器失配。也因此它在 BFM 谱系里被归入 **hierarchical control**（见上文 survey 坐标）。

### 实验与消融要点（*Science Robotics* / 中文导读归纳）

| 类别 | 结果（以 DOI 原文为准） |
|------|-------------------------|
| 用户调研 | 77 名受试者 vs Unitree 原生控制器：**70.8%** 偏好 BeyondMimic 行走/跑步拟人度；跑步 **84.7%** |
| 高动态实机 | 室外非理想地面 180° 侧手翻、连续旋踢、360° 翻转踢；腾空骨盆角速度最高 **15.7 rad/s** |
| 下游速度跟踪 | 仿真行走 / 奔跑速度误差 **12.14% / 13.65%** |
| 长距奔跑 | 真机 **50 m+** 连续奔跑；受推撞后可恢复任务 |
| 延迟 | **5 ms** 通信延迟即可失败 → 低延迟 C++ 部署栈是前提 |
| 预测视野 | 扩散策略约 **0.64 s** 前瞻，适合局部反应式控制 |

### 局限与风险

- **模式切换瞬态**：运动切换起止易踉跄；增大 classifier guidance 权重可改善任务性能但可能破坏去噪稳定性。
- **引导权重需调参**：不同任务代价的引导强度不能开箱即用。
- **无原生视觉**：障碍等信息须外部 SDF / 动捕等代价输入，不能端到端相机避障；后续可在 tracker 上做视觉后训练（如 [ViBe](../entities/paper-vibe.md) 的 LoRA 嫁接路线）。
- **上限受 RL 教师约束**：阶段 ② 扩散无法补阶段 ① 学不好的技能。
- **粗粒度代价优先**：精细动作控制弱于航点 / 速度等粗目标。

## 评价与影响

BeyondMimic 已经成为许多人形机器人项目的底层基座：
- **RobotEra (宇树春晚爆款等)**：其技术路线中大量参考了 BeyondMimic 的物理建模思想。
- **[SONIC](./sonic-motion-tracking.md)（NVIDIA/CMU 等）**：将 BeyondMimic 的能力扩展到手柄、VR 和文本控制；并被 [ExoActor](./exoactor.md) 直接当作"视频生成 → 动作估计 → 通用动作跟踪"流水线中的物理过滤器。

## 参考来源
- [机器人论文阅读笔记：BeyondMimic](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/01_Foundational_RL/BeyondMimic/BeyondMimic.html)
- [sources/papers/motion_control_projects.md](../../sources/papers/motion_control_projects.md) — 飞书公开文档《开源运动控制项目》总结。
- [sources/papers/bfm_awesome_beyondmimic_arxiv_2508_08241.md](../../sources/papers/bfm_awesome_beyondmimic_arxiv_2508_08241.md) — awesome-bfm 策展摘录（BFM 谱系坐标：05 Hierarchical control）。
- [sources/papers/loco_manip_161_survey_004_beyondmimic.md](../../sources/papers/loco_manip_161_survey_004_beyondmimic.md) — Loco-Manip 161 #004 策展摘录。
- [sources/repos/robot_lab.md](../../sources/repos/robot_lab.md) — Isaac Lab 侧集成任务与训练栈说明。
- Hybrid Robotics，[whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) — 上游开源实现与 issue 讨论入口（张量命名以仓库为准）。
- 论文：<https://arxiv.org/abs/2508.08241>（v4，2025-11-13）；正式发表：[DOI 10.1126/scirobotics.adx8924](https://doi.org/10.1126/scirobotics.adx8924)（*Science Robotics*，2026-08-26）；项目页：<https://beyondmimic.github.io/>。
- [wechat_shenlan_beyondmimic_science_robotics_2026-09-10.md](../../sources/blogs/wechat_shenlan_beyondmimic_science_robotics_2026-09-10.md) — 深蓝具身智能 *Science Robotics* 中文深度导读（实验数字、路线对照、局限；复用本页不新建实体）。
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## HMI 开源主表入口

[BeyondMimic / BeyondMimic-Reproduction](https://github.com/HybridRobotics/whole_body_tracking) 收录于具身智能研究室 [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)。

官方跟踪基线见 HybridRobotics/whole_body_tracking。主表另列社区复现 [BeyondMimic-Reproduction](https://github.com/hunter20041220/BeyondMimic-Reproduction)（教师 rollout、条件动作 VAE、扩散与测试时引导拆成可单测接口；尚未接通完整 Isaac/真机）。本库不另建复现实体，统一挂本方法页。

覆盖核对见 [HMI 开源项目主表覆盖索引](../queries/hmi-opensource-projects-coverage.md)。

## 关联页面

- [BeyondMimic（论文实体页）](../methods/beyondmimic.md) — survey 坐标（RL 身体系统栈 #15/42、BFM 地图 #35/41、Loco-Manip 161 #004）与交叉引用。
- [Imitation Learning (模仿学习)](./imitation-learning.md)
- [DeepMimic](./deepmimic.md) — 轨迹跟踪式模仿的前置脉络。
- [Armature Modeling (电枢惯量建模)](../concepts/armature-modeling.md)
- [Reward Design (奖励设计)](../concepts/reward-design.md) — 统一任务空间跟踪与分项日志的关系。
- [Curriculum Learning (课程学习)](../concepts/curriculum-learning.md) — 失败驱动采样是课程学习的一种高级形式。
- [Extreme-RGMT](../entities/paper-extreme-rgmt.md) — 同样面对高动态 vs generalist 权衡；对照表含 BeyondMimic。
- [AGILE（论文实体）](../entities/paper-agile-humanoid-loco-manipulation.md) — Isaac Lab 工作流层用 BeyondMimic 式模仿任务做案例；额外 DR+L2C2 才真机（arXiv:2603.20147）。
- [PFM-HR](../entities/paper-pfm-hr.md) — 冻结 Flow Matching 姿态几何先验；仅在仿真训练调制跟踪奖励，BeyondMimic 部署栈不变（arXiv:2608.03227；代码 Coming Soon）。
- [KDMR](../entities/paper-kdmr.md) — 用动力学可行参考替换 GMR 参考后训 BeyondMimic/mjlab 跟踪（arXiv:2603.09956）。
- [SPARK（骨架对齐重定向）](../entities/paper-spark-skeleton-aligned-retargeting.md) — KDTO(+T) 参考驱动 BeyondMimic/IsaacLab 高动态跟踪（arXiv:2603.11480）。
- [UMR](../entities/paper-umr-unified-motion-retargeting.md) — 表面对应参考喂本页跟踪协议；LAFAN1 难动作成功率高于 GMR（arXiv:2609.02134）。
