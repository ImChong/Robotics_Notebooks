---
type: query
tags: [rl, locomotion, humanoid, legged, sim2real, actuator, pd-control]
status: complete
summary: "腿足与人形 RL 中如何把 Kp/Kd（刚度/阻尼）与仿真步长、控制频率及 sim2real 对齐的工程要点与决策流程。"
updated: 2026-11-08
related:
  - ../entities/paper-digit-humanoid-locomotion-rl.md
  - ../concepts/implicit-explicit-actuator-modeling.md
  - ../entities/paper-cassie-biped-versatile-locomotion-rl.md
  - ../entities/paper-variable-stiffness-locomotion-rl.md
  - ../entities/paper-cassie-iterative-locomotion-sim2real.md
  - ../entities/paper-anymal-walk-minutes-parallel-drl.md
  - ../entities/paper-walk-these-ways-quadruped-mob.md
  - ../entities/paper-cassie-feedback-control-drl.md
  - ../entities/paper-quadruped-torque-control-rl.md
  - ../entities/paper-tan-quadruped-agile-locomotion-sim2real.md
  - ../entities/paper-variable-impedance-contact-rl.md
  - ../entities/paper-learning-quiet-walking-aibo.md
  - ../entities/paper-deeprl-locomotion-action-space-sca2017.md
  - ../entities/legged-gym.md
  - ../entities/isaac-gym-isaac-lab.md
  - ../entities/mujoco.md
  - ../concepts/sim2real.md
  - ../tasks/locomotion.md
  - ./sim2real-checklist.md
  - ./rl-hyperparameter-guide.md
  - ./domain-randomization-guide.md
original:
  - ../../sources/notes/legged_humanoid_rl_pd_gains.md
  - ../../sources/papers/learning_quiet_walking_aibo_arxiv_2502_10983.md
  - ../../sources/papers/humanoid_locomotion_reading_arxiv_2303_03381.md
  - ../../sources/courses/isaac_lab_implicit_explicit_actuators.md
  - ../../sources/papers/hwangbo_actuator_net_reading_arxiv_1901_08652_sysid1.md
sources:
  - ../../sources/notes/legged_humanoid_rl_pd_gains.md
  - ../../sources/papers/rl_pd_action_interface_locomotion.md
  - ../../sources/papers/learning_quiet_walking_aibo_arxiv_2502_10983.md
---

> **Query 产物**：本页由以下问题触发：「人形或腿足式机器人强化学习里，底层关节 Kp/Kd（或仿真里的 stiffness/damping）一般怎么设、和什么量绑定？」
> 综合来源：[legged_gym](../entities/legged-gym.md)、[Sim2Real](../concepts/sim2real.md)、[Sim2Real Checklist](./sim2real-checklist.md)、[Domain Randomization 指南](./domain-randomization-guide.md)、[开源实现索引](../../sources/notes/legged_humanoid_rl_pd_gains.md)、[RL+PD 接口论文索引](../../sources/papers/rl_pd_action_interface_locomotion.md)

# Legged / Humanoid RL 中 Kp/Kd（刚度阻尼）设置

**一句话定义**：在常见「策略输出关节目标 + 底层 PD 力矩」接口下，Kp/Kd 把策略的离散指令接到连续关节动力学上；数值必须与仿真时间步、控制分频、力矩/速度限幅以及（若做迁移）真实驱动器带宽一致考虑，而不是孤立调参。

---

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PD | Proportional–Derivative | 关节刚度/阻尼底层，RL 输出常为其 setpoint |
| Kp | Proportional Gain | 位置误差增益，影响刚度与响应 |
| Kd | Derivative Gain | 速度误差增益，抑制振荡 |
| RL | Reinforcement Learning | 策略层与 PD 层分工是 loco 常见栈 |
| Sim2Real | Simulation to Real | 增益不匹配是迁移失败常见原因 |

## 为什么和 RL 强相关

- **动作语义**：若动作为「目标关节角（或默认姿态上的残差）」，则 PD 增益直接决定同一动作向量对应的关节加速度与接触冲量，进而改变最优策略形状。
- **有效带宽**：Kp 过大易激发接触/结构高频模态（仿真里抖动、真机里电流饱和或机械共振）；过小则跟踪慢、抗扰差，reward 里速度/姿态项更难优化。
- **sim2real**：真机驱动器在电流环/速度环上等效为有限带宽；仿真里用过高刚度会学到依赖「不真实刚体响应」的技巧。域随机化常对 Kp/Kd 做区间扰动，见 [Sim2Real Checklist](./sim2real-checklist.md) 中执行器随机化条目。

---

## 各仿真栈里参数叫什么

| 栈 | 配置名 | 典型单位（关节空间） |
|----|--------|----------------------|
| legged_gym | `control.stiffness` / `control.damping`（`control_type='P'` 时 `torques = Kp·(action·action_scale + default − q) − Kd·q̇`，再按 `torque_limits` 截断；基类 `action_scale=0.5`、`decimation=4`、`sim.dt=0.005`，即策略 50 Hz） | N·m/rad，N·m·s/rad |
| mjlab | `BuiltinPositionActuatorCfg(stiffness, damping, effort_limit, armature)`；explicit 为 `IdealPdActuator` 等 | 同上；默认 `implicitfast` 积分器 |
| Isaac Lab `ImplicitActuator` | `stiffness` / `damping`（引擎内积分 PD） | 数值更稳；偏理想电机 |
| Isaac Lab `IdealPDActuator` | `stiffness` / `damping` | **explicit**：用户侧算 τ 再限幅写入仿真 |
| MuJoCo `<position>` 执行器 | `kp`、`kv`（执行器阻尼），或 `dampratio`（以 $2\sqrt{k_p m}$ 为单位，1 为临界阻尼） | 文档建议用 `implicitfast` / `implicit` 积分器 |

**Implicit vs Explicit：** 上表中的 `ImplicitActuator` 与 `IdealPDActuator` 差别不在 Kp/Kd 符号，而在 **PD 由引擎算还是用户算**——会影响收敛与 Sim2Real。详见 [Implicit / Explicit 执行器建模](../concepts/implicit-explicit-actuator-modeling.md)。

具体代码锚点与链接见 [原始资料索引](../../sources/notes/legged_humanoid_rl_pd_gains.md)。

---

## 文献地图：按子问题选读

下列论文的 **一手链接与摘录** 见 [RL+PD 动作接口与增益设计论文索引](../../sources/papers/rl_pd_action_interface_locomotion.md)。下表只保留「你关心什么 → 读哪篇」的策展顺序（与上表同一批文献）。

| 若你最关心… | 知识库子页（含 Mermaid）· 外部一手 |
|-------------|--------|
| **四种动作语义（扭矩 / 肌肉 / PD 目标角 / 目标角速度）谁更好学** — 角色侧系统对照 | [DeepRL 动作空间 SCA 2017](../entities/paper-deeprl-locomotion-action-space-sca2017.md) · [arXiv:1611.01055](https://arxiv.org/abs/1611.01055) |
| 全尺寸人形真机里 **PD 怎样嵌进大规模 RL + sim2real 流水线**（含公开增益表锚点） | [Digit 人形 RL 行走](../entities/paper-digit-humanoid-locomotion-rl.md) · [arXiv:2303.03381](https://arxiv.org/abs/2303.03381) |
| **固定标称增益 vs 随机化**、以及 **策略 Hz 与 PD 内环 Hz 的典型分频** | [Cassie 双足多技能 RL](../entities/paper-cassie-biped-versatile-locomotion-rl.md) · [arXiv:2401.16889](https://arxiv.org/abs/2401.16889) |
| **RL 能否学刚度/增益**、分组 vs 逐关节 | [可变刚度腿足 RL](../entities/paper-variable-stiffness-locomotion-rl.md) · [arXiv:2502.09436](https://arxiv.org/abs/2502.09436) |
| **为低噪触地学 PD gain scale**（触地前阻尼、支撑相加硬；消费级四足） | [Learning Quiet Walking（aibo）](../entities/paper-learning-quiet-walking-aibo.md) · [arXiv:2502.10983](https://arxiv.org/abs/2502.10983) |
| **复现落地**：reward / observation / action **多轮迭代** 的真实记录 | [Cassie 迭代式 sim2real](../entities/paper-cassie-iterative-locomotion-sim2real.md) · [arXiv:1903.09537](https://arxiv.org/abs/1903.09537) |
| **抄配置、跑消融、扫增益** 的并行仿真入口 | [ANYmal 分钟级并行 DRL](../entities/paper-anymal-walk-minutes-parallel-drl.md) · [arXiv:2109.11978](https://arxiv.org/abs/2109.11978) + [legged_gym](../entities/legged-gym.md) |
| 已能跑通固定 PD，想接 **泛化部署、行为切换与安全** | [Walk These Ways（MoB）](../entities/paper-walk-these-ways-quadruped-mob.md) · [arXiv:2212.03238](https://arxiv.org/abs/2212.03238)（**非** Walk in Minutes） |
| **为何常用 PD 目标空间** 的原始 MDP 叙述 | [Cassie 反馈控制 DRL](../entities/paper-cassie-feedback-control-drl.md) · [arXiv:1803.05580](https://arxiv.org/abs/1803.05580) |
| 判断 **是否应弃用 PD、改直驱扭矩** | [四足扭矩控制 RL](../entities/paper-quadruped-torque-control-rl.md) · [arXiv:2203.05194](https://arxiv.org/abs/2203.05194) |
| 四足 **sim2real 历史直觉**（位置/扭矩接口争论的背景） | [RSS 2018 敏捷四足 sim2real](../entities/paper-tan-quadruped-agile-locomotion-sim2real.md) · [RSS PDF p10](https://www.roboticsproceedings.org/rss14/p10.pdf) |
| **位置 + 阻抗参数联合输出** 的思想前史（对照可变刚度） | [可变阻抗接触任务 RL](../entities/paper-variable-impedance-contact-rl.md) · [arXiv:1907.07500](https://arxiv.org/abs/1907.07500) |

### 阅读与工程分支（Mermaid）

```mermaid
flowchart TB
  Q["子问题：你要优化哪一段接口"]
  Q --> A0["动作语义族对照（前史）"]
  A0 --> A01["1611.01055 SCA17 四接口"]
  Q --> H["人形全尺寸 sim2real 主干"]
  H --> H1["2303.03381 Digit RL"]
  Q --> B["双足分频与增益 DR"]
  B --> B1["2401.16889 Cassie"]
  Q --> V["刚度进动作空间"]
  V --> V1["2502.09436 Variable Stiffness"]
  Q --> I["迭代式接口设计"]
  I --> I1["1903.09537 Cassie iterative"]
  Q --> T["工程模板与并行度"]
  T --> T1["2109.11978 Walk in Minutes"]
  Q --> G["部署期行为族 / MoB"]
  G --> G1["2212.03238 Walk These Ways"]
  Q --> Y["为何 PD 目标"]
  Y --> Y1["1803.05580 Cassie feedback DRL"]
  Q --> N["不要 PD"]
  N --> N1["2203.05194 Torque RL"]
  Q --> R["四足历史参照"]
  R --> R1["RSS18 p10"]
  Q --> Z["阻抗 shaping 前史"]
  Z --> Z1["1907.07500 Var. impedance"]
```

### 策略层与 PD 内环的分频（Mermaid）

```mermaid
flowchart LR
  pol["策略网络<br/>约 20 to 50 Hz"]
  iface["动作接口<br/>q_des 或 delta q<br/>或加刚度向量"]
  pd["PD 或阻抗内环<br/>文献中可达约 2 kHz 级"]
  drv["驱动电流环<br/>更高频由硬件决定"]
  pol --> iface
  iface --> pd
  pd --> drv
  obs["本体观测"]
  obs --> pol
```

---

## 流程总览（主干）

下列流程强调 **从硬件与接口语义出发**，再落到仿真默认值与随机化范围；子步骤细节见下文分节。

```mermaid
flowchart TB
  subgraph spec["1) 语义与频率"]
    A["确认动作接口<br/>位置目标 / 速度 / 力矩"]
    B["策略频率 f_policy<br/>与仿真 dt、decimation"]
  end
  subgraph model["2) 执行器与限幅"]
    C["力矩与速度限幅<br/>齿轮比若适用"]
    D["可选：ActuatorNet / DC 电机模型<br/>替代纯 PD"]
  end
  subgraph tune["3) 仿真 Kp/Kd"]
    E["按关节组或逐关节<br/>设 stiffness / damping"]
    F["阶跃或小扰动测试<br/>无接触 → 有接触"]
  end
  subgraph sim2real["4) 迁移与鲁棒"]
    G["域随机化 Kp/Kd 区间"]
    H["与真机带宽/阻抗表对照"]
  end
  A --> B --> C --> D --> E --> F --> G --> H
```

---

## 可执行建议（精简）

1. **先锁接口与时间**：明确策略输出是 `q_des` 还是 `Δq`，以及每个策略步对应多少次仿真子步（legged_gym 的 `decimation` 即此意）。`dt` 加倍时，往往需重新评估阻尼与接触稳定性。
2. **从厂商或辨识数据取量级**：若有关节阻抗表或电流环带宽，用其倒推「位置环」上合理的 Kp 上限；也可按 Hwangbo 2019 的做法令 Kp ≈ 名义力矩范围 ÷ 名义活动范围（ANYmal：±30 N·m ÷ ±0.6 rad = 50 N·m/rad），或按 mjlab 示例由反射惯量与目标带宽推导；无数据时，从同机型开源配置（如 ANYmal-C 的 80 / 2 量级）出发，按质量和连杆长度缩放需谨慎，仅作初值。
3. **分场景试**：先在悬空或小阻尼地面做关节阶跃，检查是否饱和与振荡；再上全接触训练。振荡优先减 Kp 或增 Kd（注意二者对接触噪声的不同影响）。
4. **与 ActuatorNet 路线区分**：legged_gym 中 ANYmal 默认可走数据驱动执行器网络；此时「解析 PD」增益可能被旁路，但仍影响训练初期与 fallback 行为，需在配置层分清主路径。
5. **训练期随机化**：在 checklist 思路上，对刚度/阻尼设物理合理区间做 DR，避免策略过拟合单一阻抗点；双足文献中除 ±30% 这类宽范围外，也有 **按标称 0.7–1.3 倍缩放** 的较窄区间实践（见 [2401.16889](https://arxiv.org/abs/2401.16889) 与 [Domain Randomization 指南](./domain-randomization-guide.md) 中关节刚度条目对照）。
6. **人形公开锚点**：[2303.03381](https://arxiv.org/abs/2303.03381)（Digit）正文说明动作是 16 个驱动关节的 PD setpoint **加上 8 个腿部关节的预测 PD 增益**，4 个脚趾电机用固定增益；策略 50 Hz、关节 PD 1 kHz。2026-11-07 核对的 arXiv v2 PDF **没有**逐关节 Kp/Kd 数值表（正文把超参与观测 / 奖励 / 随机化表放在 “supplement”，该 PDF 未附），因此不要把本页或社区中的数值（如 \(K_p\approx 200\)、\(K_d\approx 10\)）当作该论文原文——**（推测）仅是社区量级传闻，需以论文补充材料或官方代码核对**。
7. **由电机规格估量级**：mjlab 文档示例由反射惯量 \(J\) 与目标带宽 \(\omega_n\) 推 \(K_p=J\omega_n^2\)、\(K_d=2\zeta J\omega_n\)（示例 10 Hz、\(\zeta=2\) 过阻尼，原因是只算了电机反射惯量、低估了总惯量），G1 示例分组：髋 180/18、膝 150/15、踝 40/5（N·m/rad，N·m·s/rad），armature 髋 0.0015、膝 0.0025、踝 0.0008。
8. **可变刚度路线**：若把刚度并入动作，优先读 [2502.09436](https://arxiv.org/abs/2502.09436) 中 **分组参数化** 与 **阻尼约束** 的设计动机，再决定仿真里是否允许 `Kd` 完全独立随机。
9. **扭矩路线门控**：若考虑去掉 PD，用 [2203.05194](https://arxiv.org/abs/2203.05194) 与当前平台的 **电流环带宽、安全滤波** 做对照实验，而不是仅改 `control_type`。

---

## 一手核对（2026-11-07）

以下数字出自本次通读的原文，用作量级参照而非推荐值：

| 来源 | 动作接口与增益 | 频率 |
|------|----------------|------|
| legged_gym 基类 | 位置目标 + 缩放 0.5；示例 Kp 10–15、Kd 1.0–1.5；ANYmal-C 配置 Kp 80、Kd 2（HAA/HFE/KFE 相同），**默认已开启 ActuatorNet（LSTM）**，解析 PD 仅在关闭开关时使用 | 仿真 dt 0.005 s，decimation 4 → 策略 50 Hz |
| Hwangbo 2019（ANYmal） | 低阻抗位置目标经固定增益转力矩：kp=50 N·m/rad、kd=0.1 N·m·s/rad、目标速度为零。选法：kp ≈ 名义力矩范围（±30 N·m）÷ 名义关节活动范围（±0.6 rad），使策略输出在位置与力矩接口下量级相近；kd 取到足以在真机上抑制振荡；作者经验是性能对增益小幅变化不敏感（kp 增到 80 无明显变化） | 部署 200 Hz（跌倒恢复 100 Hz；该技能训练于 20 Hz，提到 100 Hz 性能一致） |
| Radosavovic 2023（Digit） | 16 个关节 PD setpoint + 8 个腿关节的预测增益 | 策略 50 Hz，PD 1 kHz |
| Watanabe 2025（aibo） | 目标位置 + 每关节一个增益缩放；\(P_i=3+4\sigma(x_i)\)、\(D_i=0.03+0.02\sigma(x_i)\)；触地前增益降低（阻尼）、支撑期回升 | 仿真 400 Hz、策略 100 Hz |
| Peng & van de Panne 2017 | 2D 角色上 PD 目标角在全部运动中接近最优，7 条中 5 条学得最快；力矩动作最差；PD / 速度接口对更低查询频率（15–120 Hz 扫描）更不敏感；增益人工指定 | 主实验 60 Hz |

读法：

- **Hwangbo 的增益选法可复用，数值不可照搬**：kp 由「力矩量程 ÷ 位置量程」定标，kd 以不振荡为下限；ANYmal 的 50 / 0.1 对应 ±30 N·m、±0.6 rad 的量程，换机器人需重算。注意该论文的力矩由执行器网络给出（见 [Actuator Network](../methods/actuator-network.md)），上述固定增益描述的是策略的位置接口。
- **可变增益有收益也有代价**：aibo 上固定增益版本也能降噪但不如可变增益；整套方法鲁棒性低于普通 baseline（最大可爬坡度更小）。
- **接口决定语义**：Peng 的结论来自平面角色，3D 与真机的验证由后续工作（Digit、ANYmal、aibo）补充；论文自述超参选择会显著影响结论，趋势预期可推广但不能当定律。

## 论文共识（与调参强相关的几句）

- **接口先于增益**：同一组 `Kp/Kd`，在「输出绝对角 / 默认位姿残差 / 扭矩」下语义不同；Cassie 系列工作反复强调 **观测归一化、动作缩放与 reward 重写** 与增益表同步迭代（[1903.09537](https://arxiv.org/abs/1903.09537)）。
- **分频是设计变量**：策略频率与 PD 频率相差 **两个数量级** 在双足真机文献中很常见；把二者写进同一页系统图再调 `decimation`，比只调 `stiffness` 更接近可部署系统（[2401.16889](https://arxiv.org/abs/2401.16889)）。
- **「能学增益」≠「应全关节独立学」**：可变刚度文献表明 **结构化（分腿 / 分组）动作空间** 往往更稳（[2502.09436](https://arxiv.org/abs/2502.09436)），与工程上「按 HAA/HFE/KFE 分组」的习惯一致（参见 [legged_gym](../entities/legged-gym.md) 配置风格）。
- **历史参照**：早期四足敏捷 sim2real 证明 **强随机化 + 高频力矩/扭矩控制** 可过 gap（[RSS 2018 p10](https://www.roboticsproceedings.org/rss14/p10.pdf)），与今日「位置目标 + PD」路线并行存在，选型应回到 **带宽与安全** 而非教条。

---

## 推荐继续阅读

- Isaac Lab 执行器实现（含 Ideal / Implicit PD 说明）：<https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/actuators/actuator_pd.py>
- legged_gym 基类 PD 字段与单位注释：<https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py>
- MuJoCo XML `<position>` 执行器参考：<https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position>
- 论文索引（本页文献地图的全文链接与摘录）：[sources/papers/rl_pd_action_interface_locomotion.md](../../sources/papers/rl_pd_action_interface_locomotion.md)

---

## 参考来源

- [腿足/人形 RL 关节 PD 增益原始资料索引](../../sources/notes/legged_humanoid_rl_pd_gains.md)
- [RL+PD 动作接口与增益设计论文索引](../../sources/papers/rl_pd_action_interface_locomotion.md)
- [Learning Quiet Walking 论文归档（arXiv:2502.10983，2026-11-07 全文通读）](../../sources/papers/learning_quiet_walking_aibo_arxiv_2502_10983.md)
- [Real-World Humanoid Locomotion with RL 通读归档（arXiv:2303.03381）](../../sources/papers/humanoid_locomotion_reading_arxiv_2303_03381.md)
- [Isaac Lab / mjlab 执行器官方文档归档（含 Peng arXiv:1611.01055 摘录）](../../sources/courses/isaac_lab_implicit_explicit_actuators.md)
- [Hwangbo 2019 执行器网络论文通读归档](../../sources/papers/hwangbo_actuator_net_reading_arxiv_1901_08652_sysid1.md)

---

## 关联页面

### 论文实体子页（各含 Mermaid）

- [Digit 人形 RL 行走](../entities/paper-digit-humanoid-locomotion-rl.md)
- [Cassie 双足多技能 RL](../entities/paper-cassie-biped-versatile-locomotion-rl.md)
- [可变刚度腿足 RL](../entities/paper-variable-stiffness-locomotion-rl.md)
- [Cassie 迭代式 sim2real](../entities/paper-cassie-iterative-locomotion-sim2real.md)
- [ANYmal 分钟级并行 DRL](../entities/paper-anymal-walk-minutes-parallel-drl.md)
- [Walk These Ways（MoB）](../entities/paper-walk-these-ways-quadruped-mob.md)
- [Cassie 反馈控制 DRL](../entities/paper-cassie-feedback-control-drl.md)
- [四足扭矩控制 RL](../entities/paper-quadruped-torque-control-rl.md)
- [RSS 2018 敏捷四足 sim2real](../entities/paper-tan-quadruped-agile-locomotion-sim2real.md)
- [可变阻抗接触任务 RL](../entities/paper-variable-impedance-contact-rl.md)
- [Learning Quiet Walking（aibo 可变 PD 低噪）](../entities/paper-learning-quiet-walking-aibo.md)

### 其他

- [人形机器人 RL 策略训练完整 Checklist](./humanoid-rl-cookbook.md)
- [Sim2Real Checklist](./sim2real-checklist.md)
- [RL 超参数调节指南（locomotion）](./rl-hyperparameter-guide.md)
- [Domain Randomization 指南](./domain-randomization-guide.md)
- [MIT 关节阻抗模式参数整定](../methods/mit-joint-impedance-mode-tuning.md) — 同一 $\tau = K_p e + K_d \dot e + \tau_{ff}$ 在驱动器 MIT 帧上的手工整定（先前馈后增益、逐关节阶跃验收），可作 RL 下发 Kp/Kd 前的真机内环基线
- [Locomotion](../tasks/locomotion.md)
- [legged_gym](../entities/legged-gym.md)
- [Isaac Gym / Isaac Lab](../entities/isaac-gym-isaac-lab.md)
- [Sim2Real](../concepts/sim2real.md)
