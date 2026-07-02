# HumanoidMimicGen: Data Generation for Loco-Manipulation via Whole-Body Planning

> 来源归档（ingest）

- **标题：** HumanoidMimicGen: Data Generation for Loco-Manipulation via Whole-Body Planning
- **类型：** paper
- **来源：** arXiv / OpenReview / ICRA 2026 Workshop on Synthetic Data for Robot Learning（**Best Workshop Paper Finalist**）
- **原始链接：**
  - <https://arxiv.org/abs/2605.27724>
  - <https://arxiv.org/html/2605.27724v1>
  - OpenReview：<https://openreview.net/forum?id=ekzk7TSLKr>
  - 项目页：<https://humanoidmimicgen.github.io/>
- **机构：** NVIDIA；The University of Texas at Austin
- **入库日期：** 2026-07-02
- **一句话说明：** 将 **MimicGen 式技能片段适配** 扩展到 **双足人形 loco-manipulation**：混合动作空间（Homie RL 下肢 + 上身关节控制）与 **静态操作 / 动态行走解耦规划** 交织 **全身 IK + cuRobo 碰撞规划**，从 **少量 VR 遥操作示范** 批量合成 G1 九任务仿真数据；**GR00T N1.6 VLA** 微调平均成功率 **0.89**，sim-and-real co-training 真机平均 **+20%**。

## 核心论文摘录（MVP）

### 1) 问题：固定操作臂上的 MimicGen 无法直接用于人形

- **链接：** <https://arxiv.org/abs/2605.27724> §1–2
- **摘录要点：** VLA 需要大规模 manipulation 数据，但人形 **loco-manipulation** 遥操作成本极高；现有 **MimicGen / SkillGen / DexMimicGen / MoMaGen** 假设 **稳定、独立的 task-space 末端控制**，不适用于需 **全身协调平衡** 的双足人形（腿不能独立 OSC）。HumanoidMimicGen 目标：用 **少量专家示范** + **全身技能规划** 自动合成跨场景布局的 loco-manip 轨迹。
- **对 wiki 的映射：**
  - [HumanoidMimicGen](../../wiki/entities/paper-humanoidmimicgen.md) — 问题定义、与 MimicGen 谱系对照

### 2) 混合动作空间 + Homie 下肢 RL 控制器

- **链接：** <https://arxiv.org/abs/2605.27724> §4.1
- **摘录要点：**
  - 低层：全关节 **关节位置控制**。
  - 高层 API：**(i)** 上身（臂/手/躯干）关节目标；**(ii)** 基座运动命令 $a[l]=[\dot{x},\dot{y},\dot{\theta},z]$。
  - 下肢采用 **Homie** RL locomotion controller：输入当前/目标臂躯干配置 + 基座命令，输出 **动态可行** 腿关节位置。
  - 遥操作：Pico VR 控制器 → 末端位姿、夹爪、摇杆导航；仿真与真机同一接口。
- **对 wiki 的映射：**
  - [HumanoidMimicGen](../../wiki/entities/paper-humanoidmimicgen.md) — 混合控制空间与 teleop 协议

### 3) 技能约束 DAG + 全身数据生成主循环

- **链接：** <https://arxiv.org/abs/2605.27724> §3–4.3
- **摘录要点：**
  - 源示范按 **object-centric skill** $\psi=\langle e,f,d^\psi\rangle$ 分段，带 **precedence** $\mathcal{P}$ 与 **coordination** $\mathcal{C}$ 约束；编译为技能 DAG，贪心拓扑分组执行（如 Table-to-Shelf：先双手 pick 并发，再双手 place 并发）。
  - 每轮迭代：适配末端目标 $T[e]$ → **whole-body IK** 得 $q''$ → 构造 **switch config** $q'$（上身 $q$ + 下身 $q''$ 腿）→ **locomotion plan** $\tau_l$（RL 执行）→ **manipulation plan** $\tau_m$（上身关节控制）→ **adapt-skill-demos** 回放适配技能段。
  - 物体帧 **SE(3) 刚体变换** 适配 $a'[e]=s'[f]{s^\psi_0[f]}^{-1}a^\psi[e]$；手关节直接 replay。
  - **cuRobo**：GPU 碰撞球分解、batch IK、运动规划；接触对可 **缩小碰撞球** 避免 over-approximation 导致不可行。
- **对 wiki 的映射：**
  - [HumanoidMimicGen](../../wiki/entities/paper-humanoidmimicgen.md) — Mermaid 流程、Algorithm 1/2 归纳

### 4) 扰动策略、G1 九任务基准与仿真结果

- **链接：** <https://arxiv.org/abs/2605.27724> §4.4–5, Table 1
- **摘录要点：**
  - **Motion noise**：执行 $a+\epsilon$、标签仍存 $a$；**Init pose randomization**：基座位姿扰动。
  - **G1 Loco-Manipulation Benchmark**（robosuite + MuJoCo）：9 任务（Box Lift Floor、Push Button、Box Lift、Push Shelf Forward、Drill Lift、Drill PnP、Box Table to Shelf、Pick Drill From Holder、Drill Lift Obstacle）；沿 **基座移动量 / 交互复杂度 / 时域** 三轴变化。
  - 每任务 **1 条人类示范 → 1000 条生成轨迹**；对比 1 demo / 100 demos / **DexMimicGen+**（扩展 DexMimicGen 加直线插值 locomotion，无 skill reasoning 与碰撞规划）。
  - **HumanoidMimicGen 平均 PSR 0.89** vs DexMimicGen+ **0.33** vs 100 human demos **0.48**；长时域 Push Shelf Forward（1230 steps）**1.00** vs DexMimicGen+ **0.35**。
  - Ablation：去 motion noise **0.89→0.49**；固定初始位姿 **0.89→0.51**。
- **对 wiki 的映射：**
  - [HumanoidMimicGen](../../wiki/entities/paper-humanoidmimicgen.md) — 基准任务表、仿真实验与 ablation

### 5) 策略学习与真机 sim-and-real co-training

- **链接：** <https://arxiv.org/abs/2605.27724> §6
- **摘录要点：**
  - 仿真策略：**GR00T N1.6** VLA 微调（224×224 ego RGB + proprio），25K steps；对比 **AdaFlow** flow matching（**0.86**）与 **Diffusion Policy**（**0.51**）。
  - 真机 G1：flow matching 从零训练；**ThrowBottle / BoxToCart**（30 real + 500 sim）；**PickCanister / PickCanisterWithObstruction**（50 real ×2 variant + 1000 sim）。
  - **Co-training vs Real-only 平均 0.71 vs 0.51（+20%）**；Luxonis OAK-D，上身 25 Hz / 下身 50 Hz / 底层 200 Hz。
- **对 wiki 的映射：**
  - [HumanoidMimicGen](../../wiki/entities/paper-humanoidmimicgen.md) — 策略架构 ablation、真机任务与 co-training 结果

## 引用（arXiv BibTeX 风格）

```bibtex
@article{lin2026humanoidmimicgen,
  title={HumanoidMimicGen: Data Generation for Loco-Manipulation via Whole-Body Planning},
  author={Lin, Kevin and Mandlekar, Ajay and Garrett, Caelan Reed and
          Chernyadev, Nikita and Fang, Yu and Ding, Runyu and Xie, Yuqi and
          Tran, Justin and Fan, Linxi and Zhu, Yuke},
  journal={arXiv preprint arXiv:2605.27724},
  year={2026}
}
```
