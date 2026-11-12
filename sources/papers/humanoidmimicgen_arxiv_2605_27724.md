# HumanoidMimicGen: Data Generation for Loco-Manipulation via Whole-Body Planning

> 来源归档（ingest）

- **标题：** HumanoidMimicGen: Data Generation for Loco-Manipulation via Whole-Body Planning
- **类型：** paper
- **来源：** arXiv / OpenReview / ICRA 2026 Workshop on Synthetic Data for Robot Learning（**Best Workshop Paper Finalist**）
- **原始链接：**
  - <https://arxiv.org/abs/2605.27724>
  - <https://arxiv.org/pdf/2605.27724v1>（本次通读版本，v1，2026-05-26，cs.RO）
  - <https://arxiv.org/html/2605.27724v1>
  - OpenReview：<https://openreview.net/forum?id=ekzk7TSLKr>（本库原记录，本环境打不开，未核对）
  - 项目页：<https://humanoidmimicgen.github.io/>
  - 代码：<https://github.com/NVlabs/humanoidmimicgen>（项目页按钮所列；本次未能访问该仓库，内容与开放程度未核实）
- **机构：** NVIDIA；The University of Texas at Austin
- **入库日期：** 2026-07-02（2026-11-12 升级为全文通读版）
- **核对范围：** 通读 arXiv v1 PDF 全文：正文 §1–8（含 Fig.1–5、Table 1）、参考文献、附录 A–I（含 Algorithm 1/2、Table 2–5、真机评分细则）；另打开项目页（作者、链接按钮）与 arXiv 摘要页。未读：项目页视频与图、GitHub 仓库内容（环境无权限访问）、被引工作（MimicGen、DexMimicGen、Homie、GR00T N1 等）原文。论文未出现 Best Workshop Paper Finalist 或 ICRA workshop 字样，本库此前该说法在 arXiv 论文与项目页均未核实到。
- **通读：** 已完成
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

### 6) 补充：全文核对新增细节（2026-11-12）

- **链接：** <https://arxiv.org/pdf/2605.27724v1>
- **摘录要点：**
  - §4.1：基座命令 $a[l]=[\dot x,\dot y,\dot\theta,z]$，其中 $\dot x,\dot y$ 为平面速度、$\dot\theta$ 为偏航率、$z$ 为期望躯干高度。
  - §4.2：每个末端执行器同一时刻只能做一个技能，故同时活跃技能至多 $|E|=2$ 个。
  - §4.3 / Alg.1：IK 返回一批 $Q$，逐个尝试 locomotion 规划，批次用尽则丢弃该回合；数据集靠反复调用直到成功 $N$ 次。生成时假设可访问特权状态，策略只用观测（§3）。
  - §4.3：手关节 $a^\psi[J_h]$ 原样回放，不做适配。
  - 附录 C：碰撞球由网格表面采样点求最小体积切球后加 $\epsilon\approx0.01$ m 膨胀、贪心选 $N$ 个；IK 用 L0 距离偏好，按自由关节顺序 [臂, 臂+躯干, 臂+腿, 全部] 迭代；若不动腿即可到达则跳过 locomotion 规划。
  - 附录 D / Table 2：9 任务按 Loco、Nav、1-arm、2-arm、Vert、Contact、Long 标注能力。
  - 附录 E：DexMimicGen+ 沿用同一动作空间，人工标注需行走的子任务，用不考虑碰撞的 IK 推基座目标位姿，直线插值，无运动规划。
  - 附录 F / Table 3：去腿的浮动基座人形（G1 w/o Legs）平均 PSR 0.90，对比 G1 的 0.89；逐任务差异大（Push Shelf Forward 1.00→0.53，Box Table to Shelf 0.53→1.00）。
  - 附录 G：GR00T N1.6 微调学习率 1e-5、全局 batch 128、25,000 步、8×H100；每 5,000 步评估一次，每任务 100 次 rollout，报告**最佳检查点**。
  - 附录 H / Table 5：去 motion noise 后 Box Lift Floor 与 Push Button 反为 1.00；平均下降主要来自 Drill Lift（0.23）、Drill PnP（0.17）、Box Table to Shelf（0.00）、Pick Drill from Holder（0.26）、Drill Lift Obstacle（0.17）。
  - 附录 I：真机每策略评估 20 回合；评分为部分得分制（BoxToCart：抬箱 0.5、走到车 0.25、放上车 0.25；ThrowBottle：抬瓶 0.5、投入桶 0.5；PickCanister 系：走到货架 0.5、抬罐 0.5）；上身 25 Hz、下身策略 50 Hz、底层位置控制插值 200 Hz。
  - §6.4：ThrowBottle / BoxToCart 为 30 条真机 + 500 条仿真；PickCanister 两个变体共训一个策略，各 50 条真机 + 1000 条仿真。
- **对 wiki 的映射：**
  - [HumanoidMimicGen](../../wiki/entities/paper-humanoidmimicgen.md) — 实验细节与「与清单说法的核对」

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
