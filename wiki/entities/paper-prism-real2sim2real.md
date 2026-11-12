---
type: entity
tags:
  - paper
  - humanoid
  - loco-manipulation
  - real2sim2real
  - video-generation
  - depth-policy
  - contact-rich-manipulation
  - amazon
  - berkeley
  - stanford
  - cmu
status: complete
updated: 2026-11-12
arxiv: "2609.38172"
venue: "CoRL 2026"
related:
  - ../methods/crisp-real2sim.md
  - ../concepts/sim2real.md
  - ../tasks/loco-manipulation.md
  - ./videomimic.md
  - ./paper-resmimic.md
  - ./paper-light-loco-parkour.md
  - ../methods/dagger.md
  - ../concepts/motion-retargeting.md
  - ../../roadmap/depth-real2sim.md
  - ./omomo-dataset.md
  - ./paper-hrl-stack-03-omniretarget.md
  - ./paper-hrl-stack-06-hdmi.md
original:
  - ../../sources/papers/prism_real2sim2real_arxiv_2609_38172.md
sources:
  - ../../sources/papers/prism_real2sim2real_arxiv_2609_38172.md
  - ../../sources/sites/prism-real2sim2real-github-io.md
  - ../../sources/repos/prism-real2sim2real.md
summary: "PRISM（arXiv:2609.38172，Amazon FAR×Berkeley×Stanford×CMU，CoRL 2026）：V2V counterfactual 把 4 条搬箱视频扩成 256 条，137 条可重定向、129 条成功 teacher rollout，接触锚定 Real2Sim+重定向训深度+摇杆 G1 策略，真机零样本 pick–carry–drop（in-domain 80–100%，OOD 每类仅 5 次试验）；仿真对比用的是更早的 80 条版 student。"
---

# PRISM（Counterfactual V2V Real2Sim2Real Loco-Manipulation）

> **命名消歧：** 本页为 **Real2Sim2Real + counterfactual 视频** 的 PRISM（[arXiv:2609.38172](https://arxiv.org/abs/2609.38172)）。同缩写但不同工作的 **[多项式本体表征 PRISM](./paper-prism.md)** 见 arXiv:2607.23473；VLA 侧 **[Prism-GRPO](./paper-prism-grpo.md)** 见 arXiv:2608.17423。

## 一句话定义

**PRISM**（*Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation*，CoRL 2026，[项目页](https://prism-real2sim2real.github.io/)）：用 video-to-video（V2V）生成把 4 条真人搬箱视频扩成 256 条「换物体的反事实交互」，经接触锚定的 Real2Sim 与重定向得到 G1–物体轨迹，训练只靠机载深度加摇杆的统一全身策略，真机零样本完成 pick–carry–drop，无真机微调（Abstract、§1）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PRISM | （本文系统名） | Real2Sim2Real + counterfactual V2V loco-manip 框架 |
| V2V | Video-to-Video | 以 seed 视频加类别 prompt 为条件的生成，保留背景、光照、机位（§3.1） |
| R2S2R | Real-to-Sim-to-Real | 真视频 → 仿真演示 → 真机零样本 |
| CF | Counterfactual | 源视频中未发生但换个物体就可能发生的交互 clip |
| CRISP | Contact-guided Real2Sim from monocular video | 单目人–场景–相机恢复后端，本文扩展到动态物体（§3.2） |
| HMR | Human Mesh Recovery | 人体网格与姿态估计，输出 SMPL-X 4D 运动 |
| DAgger | Dataset Aggregation | 与 PPO 混合的蒸馏损失（式 3） |
| G1 | Unitree G1 | 29-DoF 真机，50 Hz 部署（§5.2） |

## 为什么重要

- **数据瓶颈换思路（作者主张）：** 不去互联网筛清晰的全身交互片段，而是让视频生成模型在真实 seed 上做 grounded 扩增（§1、§2「Video Model as Data++」）。
- **接触作为系统纽带：** 同一组接触锚点约束物体位姿优化、重定向 IK 和 RL 接触奖励，用来消化 V2V 与单目重建的误差（§1、§3.2、§4.1）。表 3 里三步逐级把 PRISM-ID 从 22.50% 抬到 96.25%。
- **部署口径干净：** 深度 + 摇杆 + 单策略，部署时无 MOCAP、无参考动作（§4.2）；对照需要稠密参考动作或物体轨迹的跟踪式系统（§2）。
- **论文自称**是首个训练统一全身策略、用机载深度和摇杆完成多物体 pick–carry–drop 的 R2S2R 管线（§1 贡献 3）；这是作者的新颖性表述，本页未独立核实。

## 核心原理

### 三段流水线

| 阶段 | 输入 / 输出 | 要点（出处） |
|------|-------------|--------------|
| Counterfactual V2V | 4 条真人搬箱 seed → 256 条 CF clip | SeedDance 2.0 网页界面，每 seed × 4 类 × 16 样本；prompt 仅一句模板（§3.1、附录 A） |
| 接触锚定 Real2Sim | CF 视频 → 人体 + 静态场景 + 物体 6D | CRISP 后端 + SAM 2 掩码 + SAM3D 网格；接触相位内物体固定在掌坐标系（§3.2） |
| 接触锚定重定向 | SMPL-X + 物体 → G1–物体轨迹 | OmniRetarget 式 interaction mesh IK 加末端到锚点项（式 1） |
| 策略学习 | 轨迹 → 深度 student | 特权 co-tracking teacher + 接触奖励（式 2），DAgger+PPO 蒸馏（式 3）（§4） |

### 流程总览

```mermaid
flowchart LR
  seed["4 条真人 seed 视频"]
  v2v["SeedDance 2.0 V2V<br/>256 条 counterfactual"]
  r2s["接触锚定 Real2Sim<br/>CRISP + SAM2 + SAM3D"]
  ret["接触锚定重定向<br/>137 条可行轨迹"]
  teach["Co-tracking teacher<br/>129 条成功 rollout"]
  stud["深度 + 摇杆 student<br/>DAgger 加 PPO"]
  real["G1 真机<br/>零样本 pick-carry-drop"]
  seed --> v2v --> r2s --> ret --> teach --> stud --> real
```

### 数据漏斗（附录 C、§7）

256 条生成视频中只有 137 条得到可行的机器人–物体轨迹（求解器失败主因是严重碰撞，部分生成物体对 G1 过大），teacher 在其上训练后得到 129 条成功 rollout，这 129 条才是 student 的演示。「4 seed 扩成 256 条」不等于 256 条训练数据。

### 源码运行时序图

**不适用**：论文称训练细节见其代码库（§4.1），但项目页链接的 [`amazon-far/PRISM-Real2Sim2Real`](https://github.com/amazon-far/PRISM-Real2Sim2Real) 在 2026-09-30 入库时匿名访问为 404，2026-11-12 本次经代理访问返回 403，仍无法确认公开状态，因此无源码可对齐，不画 `sequenceDiagram`。

## 工程实践

| 项 | 论文给出的做法（出处） |
|----|------------------------|
| Seed 选择 | 稳定视角、少遮挡、任务兼容动作；V2V 只需简单 prompt（§5.3） |
| 深度 Sim2Real | 离板 Fast-FoundationStereo 估深度，作者称显著缩小深度 gap；训练中随机化相机并注入噪声、丢点、空洞、偏移（§4.2、§5.2、表 6） |
| 相机 / 颈部 | 手动把颈部上仰约 10°，依赖相机位姿随机化容忍标定误差；弯腰时颈部偶尔复位或偏离，造成 OOD 深度（§5.2） |
| 暖启动 | 先用 box-only 训 23K iter，只恢复 actor 权重再训全类；从零训练也成功（§4.2、附录 D） |
| 训练算力 | Teacher 40K、student 28K iter，8 张 L40S × 4096 环境，3 层 MLP（§4.2） |
| 物体物理 | 类别名义质量（球 0.5、箱和筐 1.0、桶 1.5 kg）乘 U(0.33, 3.0) 缩放，摩擦随机（附录 E） |
| 仿真吞吐 | Isaac Lab 里异构资产限制仿真与训练吞吐（§5.3） |
| 管线耗时 | 每 8 秒 clip：V2V 191.76 s、重建 414.5 s、重定向 71.6 s（附录 G.2，表 10） |

## 实验与评测

### 跨域（表 1，MuJoCo sim-to-sim，更早的 80 条版 student）

| 训练数据 | OMOMO Test | PRISM-ID | PRISM-OOD |
|----------|-----------|----------|-----------|
| OMOMO (90) | 91.67% | 23.75% | 12.50% |
| PRISM-ID (80) | 100% | 96.25% | 72.92% |

两点口径：PRISM-ID 就是那 80 条训练交互，不是留出集；表 1 用的是由 20K iter teacher 的 80 条 rollout 蒸馏出的更早 student，更新后 129 条版本不在此比较里（附录 C）。PRISM-OOD 是 48 条重建交互（table / chair / lamp / monitor 各 12 个实例）。

### 真机（表 2，每物体 5 次试验）

| 类别 | Ball | Bin | Barrel | Box |
|------|------|-----|--------|-----|
| in-domain（每类 3 个物体，共 15 次） | 12/15 | 14/15 | 14/15 | 15/15 |

OOD（每类 1 个物体 × 5 次）：纸巾 5/5、桌子 5/5、背包 5/5、椅子 4/5、水壶 4/5、头盔 3/5、灯 3/5、小鸡玩具 3/5。成功判据是够到抓起并稳定搬运至少 3 m 不掉不倒，不含放置到目标位置，与仿真判据不同（§5.2、附录 C）。

### 消融（表 3，PRISM-ID / PRISM-OOD）

| 栈 | ID | OOD |
|----|-----|-----|
| Baseline（SAM3D + FoundationPose + OmniRetarget） | 22.50% | 12.50% |
| + 接触锚定物体位姿 | 58.75% | 29.17% |
| + 接触感知重定向 | 86.25% | 68.75% |
| + 接触奖励（完整 PRISM） | 96.25% | 72.92% |

### V2V 对几何增广（附录 F，表 9）

用 8 条 seed 的口径：8 条原始 seed 演示 → ID/OOD 17.50/6.25；加 yaw/xy/scale 变体的 103 条 → 36.25/22.92；V2V 80 条 → 96.25/72.92。这一比较用的仍是 80 条版 student，不评估当前 4 seed、129 rollout 配置。

### 其他

- 零样本高处拾取：35° 斜坡与 0.43 m 平台成功，45° 与 0.45 m 失败，作者推测与训练物体抓取高度重叠有关（§5.3、图 5）。
- 载荷敏感度（仿真、80 条版，表 8）：PRISM-ID 从 0.1–1.0 kg 的 98.75% 降到 4–5 kg 的 91.25%；PRISM-OOD 从 77.08% 降到 64.58%，中间非单调。

## 与清单说法的核对

清单（Glen Berseth 的 CoRL 2026 阅读清单第 053 条，P1 档，分组 Learning from Human Video，据摘要由 LLM 起草）的一句话：

> Amplifies a few human videos into many counterfactual interaction videos, retargeted through real-to-sim, to train humanoid loco-manipulation.

| 清单措辞 | 论文核对结果 |
|----------|--------------|
| a few human videos | 成立：4 条真人搬箱 seed（§1、附录 A）；但消融另用 8 条 seed（§5.3、附录 F），论文未统一两种口径 |
| many counterfactual interaction videos | 生成 256 条，但只有 137 条得到可行轨迹、129 条成为 student 演示（附录 C、§7）；作者自己说规模不足以研究 scaling |
| retargeted through real-to-sim | 成立，且是论文重点：接触锚定重建加接触锚定重定向，消融是主要证据（表 3） |
| to train humanoid loco-manipulation | 成立但范围窄：只有 pick–carry–drop，物体限刚体，策略依赖摇杆；清单摘要没提「机载深度」「真机零样本」，这两点才是论文的部署结论 |

清单排序是作者按自己兴趣模型打分得到的，不是论文质量评价；上述核对只对摘要句，不对档位作判断。

## 局限

论文自述（§7、附录）：

- 单目 4D 人–物–场景重建仍难，误差沿重定向传播；接触锚无形式化可行性保证（附录 G.1）。
- 数据规模不足以刻画 scaling（4 → 256 → 137 → 129）。
- 只用类别名义质量与随机摩擦，未辨识实例属性；刚体假设下折叠椅等关节物体会不稳。

我读到的未验证之处（判断，非论文自述）：

- 表 1、表 9 的 PRISM-ID 是训练交互本身，且来自更早的 80 条版 student；与真机用的 129 条版 student 并非同一配置，仿真数字不能直接外推到真机。
- 对几何增广的对比评测集由 V2V 数据构成，对对照方有分布偏向（推测）。
- 真机 OOD 每类仅 1 个物体 × 5 次试验，60–100% 的区间置信度很宽；in-domain 也只有每类 15 次。
- 真机判据不含放置到目标，仿真判据含放置，两者不可直接比较。
- 「人的行为随物体适配」是作者对生成视频的描述，论文未量化；V2V 引入的不物理动作靠 137/256 的可行性过滤间接剔除。
- 深度靠离板 Fast-FoundationStereo，非机上自洽；操作员摇杆给方向。
- 与 [paper-prism.md](./paper-prism.md)、[paper-prism-grpo.md](./paper-prism-grpo.md) 缩写碰撞，引用时带 arXiv 或全称。

## 关联页面

- [CRISP Real2Sim](../methods/crisp-real2sim.md) — 单目人–场景恢复后端，同第一作者的 CRISP 线
- [Loco-Manipulation 任务](../tasks/loco-manipulation.md)
- [Sim2Real 概念](../concepts/sim2real.md)
- [Motion Retargeting](../concepts/motion-retargeting.md) · [OmniRetarget 论文页](./paper-hrl-stack-03-omniretarget.md) — 重定向基线
- [DAgger](../methods/dagger.md) — 蒸馏损失来源
- [OMOMO 数据集](./omomo-dataset.md) — 表 1 的对照数据
- [VideoMimic](./videomimic.md) · [ResMimic](./paper-resmimic.md) · [HDMI](./paper-hrl-stack-06-hdmi.md) — 视频 / 残差 / 交互式人形 loco-manip 对照
- [Light-Loco-Parkour](./paper-light-loco-parkour.md) — 深度 + 速度的人形策略对照
- [Real2Sim 纵深 Stage 4](../../roadmap/depth-real2sim.md)

## 参考来源

- [PRISM 论文通读归档](../../sources/papers/prism_real2sim2real_arxiv_2609_38172.md)
- [PRISM 项目页归档](../../sources/sites/prism-real2sim2real-github-io.md)
- [PRISM-Real2Sim2Real 仓库索引](../../sources/repos/prism-real2sim2real.md)

## 继续阅读

- [项目页](https://prism-real2sim2real.github.io/) — 真机物体矩阵与 robustness 视频
- [arXiv:2609.38172](https://arxiv.org/abs/2609.38172)
- [CRISP（ICLR 2026）方法页](../methods/crisp-real2sim.md)
- [Z1hanW/CRISP-Real2Sim（GitHub）](https://github.com/Z1hanW/CRISP-Real2Sim) — 同作者 Real2Sim 开源基线
