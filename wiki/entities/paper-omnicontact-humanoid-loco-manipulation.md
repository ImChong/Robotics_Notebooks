---
type: entity
tags:
- paper
- humanoid
- loco-manipulation
- reinforcement-learning
- motion-tracking
- contact-flow
- meta-skill
- sim2sim
- mocap
- unitree-g1
- vlm
- noitom
- hkust
- hku
- whu
- repo
- mujoco
- omnicontact
status: complete
updated: 2026-11-21
arxiv: '2606.26201'
venue: CoRL 2026（arXiv v1: 2026-06-24）
related:
- ../tasks/loco-manipulation.md
- ../concepts/contact-rich-manipulation.md
- ../concepts/whole-body-control.md
- ../concepts/sim2real.md
- ../concepts/motion-retargeting.md
- ../methods/amp-reward.md
- ../methods/vla.md
- ./paper-amp-survey-15-physhsi.md
- ./paper-hrl-stack-06-hdmi.md
- ./paper-notebook-lessmimic-long-horizon-humanoid-interaction-with.md
- ./paper-hrl-stack-05-humanx.md
- ./paper-hrl-stack-03-omniretarget.md
- ./paper-umr-unified-motion-retargeting.md
- ./unitree-g1.md
- ./humanoid-gym.md
- ./paper-notebook-visualmimic.md
sources:
- ../../sources/papers/omnicontact_arxiv_2606_26201.md
- ../../sources/sites/omnicontact-project.md
- ../../sources/repos/omnicontact-sim2sim.md
- ../../sources/repos/omnicontact-training.md
- ../../sources/datasets/omnicontact-dataset.md
original:
- ../../sources/papers/omnicontact_arxiv_2606_26201.md
summary: OmniContact（arXiv:2606.26201，诺亦腾/HKUST/WHU/HKU）以 Contact Flow（稀疏体目标+四端接触时序）统一 CF-Track 低层 RL 与 CF-Gen 规则重规划，实现 meta-skill 链式组合、50 Hz 自主恢复与 VLM 语义任务；Carry Box 98.7%、Push-Stack 76.5%，配套 22.29 h MoCap 数据集与 MuJoCo sim2sim。
project_id: omnicontact-humanoid-loco-manipulation
code: https://github.com/Ingrid789/OmniContact_sim2sim
project: https://omnicontact.github.io/
---

# OmniContact（Chaining Meta-Skills via Contact Flow）

**OmniContact**（*Chaining Meta-Skills via Contact Flow for Generalizable Humanoid Loco-Manipulation*，arXiv:2606.26201，[项目页](https://omnicontact.github.io/)）提出以 **Contact Flow（CF）** 为规划–执行共享接口的分层 loco-manipulation 框架：**CF-Track** 在统一 RL 策略下跟踪多样 meta-skill，**CF-Gen** 用物体几何规则合成 CF 并在 **50 Hz** 闭环重规划，实现长时程 **skill chaining**、**自主 recovery** 与 **VLM 语义分解**；配套自采 **OmniContact MoCap HOI 数据集**（1,274 序列 / 22.29 h）与 [MuJoCo sim2sim 仓库](https://github.com/Ingrid789/OmniContact_sim2sim)；CF-Track [Isaac Lab 训练代码](https://github.com/Ingrid789/OmniContact)也已开放。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CF | Contact Flow | 稀疏关键体轨迹 + 四端二进制接触信号的 skill 中间表示 |
| HOI | Human-Object Interaction | 人–物交互；本文强调 90 Hz 同步 MoCap + 6-DoF 物体轨迹 |
| CF-Track | Contact Flow Track | 统一低层 RL 执行器，跟踪 CF 或 NPZ 全轨迹 |
| CF-Gen | Contact Flow Generation | 规则相位模板 + 物体锚点 IK + 在线重规划的中层合成器 |
| RL | Reinforcement Learning | CF-Track 训练范式；奖励含跟踪、AMP 先验与动作正则 |
| AMP | Adversarial Motion Prior | 对抗运动先验，约束自然人形动力学 |
| VLM | Vision-Language Model | 高层语义任务分解为 start–goal 物体位姿 |
| G1 | Unitree G1 Humanoid | 论文与开源栈默认 29-DoF 人形平台 |
| NPZ | NumPy Archive | HF 数据集与 sim2sim 中的 G1 重定向轨迹格式 |

| FSM | Finite State Machine | Passive→DefaultPose→LocoMode→OmniContact 状态链 |
| ONNX | Open Neural Network Exchange | 导出的 CF-Track 策略推理格式 |

## 为什么重要

- **填补「可组合 skill 接口」空白**：dense HOI tracking（HDMI、OmniRetarget）难在线编辑；implicit embedding（LessMimic）难解释与结构化链接；OmniContact 用 **显式接触时序 + 稀疏体目标** 同时服务低层跟踪与中层启发式规划。
- **长时程闭环证据与边界**：Stack Boxes、Push-Stack 等多阶段任务基线接近 **0%**，本文 chaining 达 **56.5–76.5%**；在线重规划进一步抬升 Push/Stack 成功率。附录 C.4 的 **40 min** 测试依任务不同：单物体连续目标存活率 **100%**，五物体连续目标仅 **10%**，不可混为一谈。
- **数据与代码入口更完整**：[HuggingFace 数据集](https://huggingface.co/datasets/lightcone02/OmniContact-Dataset)（917 G1 NPZ + 原始 MoCap）、[CF-Track Isaac Lab 训练仓库](https://github.com/Ingrid789/OmniContact)（PPO/AMP、MLP/Transformer）、[sim2sim 仓库](https://github.com/Ingrid789/OmniContact_sim2sim)（CFgen/NPZ 双路径 + Xbox FSM）、项目页 **MuJoCo WASM + ONNX** 在线 viewer。复现仍需自行配置资产与运行环境。
- **VLM 即插即用**：无需改低层权重，VLM 输出物体布局即可驱动 CF-Gen 合成 CF——语义任务（心形排箱、字母拼摆、进球）与接触物理层解耦。

## 流程总览

```mermaid
flowchart TB
  subgraph high [高层可选]
    vlm["VLM 语义分解\n→ start/goal 物体位姿"]
    goal["任务目标 + 物体状态"]
    vlm --> goal
  end
  subgraph mid [CF-Gen 50 Hz]
    phase["相位模板库\napproach / grasp / carry / release…"]
    ik["物体几何锚点 + 约束 IK\nwrist/ankle 关键帧"]
    interp["LERP/SLERP 插值 → dense 参考"]
    sample["非均匀采样 T 偏移\n→ Contact Flow F_t"]
    monitor["观测–预测偏差 δ_t\n超阈则 abort 重规划"]
    goal --> phase --> ik --> interp --> sample
    sample --> monitor
  end
  subgraph low [CF-Track]
    obs["x_t = [F_t, H_t]\n本体 + 物体 6D + 历史"]
    policy["统一 RL 策略\n跟踪 + AMP + 正则"]
    act["关节力矩/位置命令"]
    obs --> policy --> act
  end
  sample --> obs
  monitor -->|"重规划"| phase
  act --> g1["Unitree G1\nMuJoCo / 真机"]
```

## 核心机制（归纳）

### 1）Contact Flow 表示

每控制步 $F_t = \{(b_{t+k}, c_{t+k})\}_{k \in \{0,1,2,3,4,8,12,16,24,32,50\}}$：

- $b_{t+k}$：稀疏 body targets（wrists、torso、ankles），在 torso-yaw 帧表达。
- $c_{t+k} \in \{0,1\}^4$：左踝、右踝、左腕、右腕 **二进制接触**。

相对 dense HOI 全轨迹：**更可编辑、可在线合成**；相对纯 object goal：**保留接触时序与交互意图**。Ablation 表明显式 contact 位将 Carry Box 从 **11.5%→98.7%**。

### 2）CF-Track：统一低层执行

| 组件 | 设计 |
|------|------|
| 输入 | 目标 CF $F_t$ + $K=5$ 步历史（关节、基座、EE、物体相对 pose、bbox、$a_{t-1}$） |
| 输出 | 低层 motor actions |
| 训练数据 | OmniContact MoCap→contact flow 监督 |
| 奖励 | $r^{\text{track}} + r^{\text{amp}} + r^{\text{reg}}$；track:amp = **0.85:0.15** 为最优平衡 |
| 能力 | 单一策略覆盖 carry / push / slide / kick / relocate；推理可跟 CF-Gen 或 NPZ 全轨迹 |

### 3）CF-Gen：相位模板 + 几何锚点

- 每 meta-skill 为 **有序 phase blocks**（如无接触接近 → 预抓 → 抬升 → 持物行走 → 释放）。
- 相位结束态由 **object-centric geometry** 锚定；接触相位用 **约束 IK**（pelvis 高/俯仰 + 关节，锁 waist roll/yaw）求 wrist/ankle。
- 相位间插值得 dense  kinematics，再采样为 CF 供 CF-Track——**不把 dense 关节命令直接喂策略**，保留低层柔顺与自然性。
- **50 Hz 监控**：$\delta_t = d(x^{\text{obj}}_{t,\mathrm{obs}}, x^{\text{obj}}_{t,\mathrm{pred}}) > \epsilon$ 时重规划 → 掉箱重接近等 recovery。

### 4）OmniContact 数据集

| 指标 | 数值 |
|------|------|
| 有效序列 | 1,274（论文）；HF 公开 917 条 G1 NPZ |
| 总时长 | 22.29 h |
| 物体帧 | 7.22M |
| 同步 | 90 Hz 人–物配对 MoCap |
| 原语 | carry / push / relocate / slide / kick |
| 平均序列 | 62.98 s；平均物体路径 19.76 m |

相对 OMOMO：更偏 **长时程搬运** 与 **contact-flow 监督**；OMOMO 序列更多、物体类别更广但 clip 更短。

## 评测（论文报告，仿真）

### Meta-skill（Table 2）

| 任务 | $R_{\text{succ}}$ | $E_{\text{obj}}^T$ (m) | $N_{\text{hoi}}$ |
|------|-------------------|------------------------|------------------|
| Carry Box | **98.7%** | 0.07 | 7.75 |
| Push Suitcase | **82.5%** | 0.27 | 6.00 |

### Meta-skill chaining

| 任务 | 阶段成功率 | 备注 |
|------|------------|------|
| Stack Boxes | 89 / 87 / **56.5%** | 三阶段 gather–stack |
| Push-Stack Boxes | 91.5 / **76.5%** | push + stack 组合 |

基线 Sonic / HDMI / PhysHSI / LessMimic 在长时程 chaining 上 **0% 或近零**；相对基线平均提升 meta-skill **+40.9%**、chaining **+66.5%**。

### 在线重规划（Table 3）

| 任务 | 无重规划 | 有重规划 $R_{\text{succ}}^*$ |
|------|----------|------------------------------|
| Push Suitcase | 82.5% | **94.5%** |
| Stack Boxes | 56.6% | **80.5%** |
| Push-Stack | 76.5% | **84.5%** |

### 长时程耐力评测边界（Appendix C.4，40 min）

| 协议 | 场景 | 40 min 存活率 |
|------|------|---------------|
| I | 同一物体的连续目标 | **100.0%** |
| II | 单物体初末目标重采样 | **29.5%** |
| III | 五个不同物体的连续目标 | **10.0%** |

上述均为论文报告的仿真耐力协议，不宜推广为真实仓储任务的可靠性指标；主 benchmark 为 5 个随机种子、每种子 200 个随机任务配置（附录 E.1）。

## 结论

**用可编辑的 Contact Flow（稀疏体目标 + 四端接触时序）同时服务低层统一跟踪与中层规则重规划，是长时程 loco-manip skill chaining 的可复现接口。**

1. **显式接触位极关键** — ablation 将 Carry Box 从 11.5% 提到 98.7%；相对 dense HOI 更可在线合成，相对纯 object goal 保留交互意图。
2. **CF-Track 单策略覆盖多 meta-skill** — carry/push/slide/kick/relocate；track:amp = 0.85:0.15 为文中最优奖励平衡。
3. **CF-Gen + 50 Hz 重规划抬升长程** — Push/Stack 等在有重规划时成功率再升（如 Push Suitcase 82.5%→94.5%，Stack 56.6%→80.5%）。
4. **chaining 相对基线近零 → 数十百分点** — Stack Boxes 终段 56.5%、Push-Stack 76.5%；相对基线平均 meta-skill +40.9%、chaining +66.5%。
5. **VLM 只出物体布局** — 语义与接触物理解耦，无需改低层权重即可驱动 CF-Gen。
6. **复现入口** — 22.29 h MoCap、HF G1 NPZ、CF-Track Isaac Lab 训练及 MuJoCo sim2sim；论文附录 D.1 的 G1 真机部署依赖外部动捕提供机器人/物体全局位姿，不能直接推断为纯机载自主系统。

## 与相邻路线对比

| 维度 | OmniContact | PhysHSI | HDMI | LessMimic |
|------|-------------|---------|------|-----------|
| Skill 表示 | **Contact Flow** | Object goal | Dense HOI TT | Skill embedding |
| 统一策略 | ✓ 单 CF-Track | ✗ 分任务 | ✓ 单轨迹易过拟合 | ✗ |
| Skill chaining | ✓ CF-Gen 链式 | ✗ | ✗ | 部分 |
| Recovery | ✓ 50 Hz 重规划 | ✓ | ✗ | ✓ |
| VLM 集成 | ✓ 物体布局→CF | ✗ | ✗ | ✗ |
| 数据 | 自采 22.29 h MoCap | AMASS+标注 | 视频→HOI | 多源 HOI |

## 开源与 sim2sim

- **CF-Track 训练**：[Ingrid789/OmniContact](https://github.com/Ingrid789/OmniContact) — Isaac Sim 4.5.0 / Isaac Lab 2.1.1、RSL-RL PPO/AMP，支持 MLP/Transformer；入口 `scripts/rsl_rl/train.py`、`scripts/rsl_rl/play.py`，有预训练 checkpoint 下载入口。见 [训练代码来源归档](../../sources/repos/omnicontact-training.md)。
- **仓库**：[Ingrid789/OmniContact_sim2sim](https://github.com/Ingrid789/OmniContact_sim2sim) — CFgen / NPZmotion 双参考源、`carry-push` 等 chaining preset、Xbox FSM 热切换（镜像 sim2real）。
- **数据集**：[lightcone02/OmniContact-Dataset](https://huggingface.co/datasets/lightcone02/OmniContact-Dataset) — G1 NPZ + BVH 原始 MoCap + Viser 可视化。
- **在线 demo**：项目页 MuJoCo WASM + ONNX policy viewer。
- **sim2sim 模型名**：官方 README 提供 `omnicontact_transformer.onnx`（默认）和 `omnicontact_mlp.onnx`，均位于 `policy/omnicontact/model/`；勿沿用旧的 `policy.onnx` 示例名。

详见实体页 [OmniContact sim2sim](#项目资源与工程补充)（仓库工具页）。

## 常见误区或局限

- **不是端到端 VLA 全身策略**：VLM 只产出 **物体级布局**；接触与动力学仍由 CF-Gen + CF-Track 承担——语义层与物理层刻意解耦。
- **CF-Gen 仍为规则/heuristic**：对高度动态场景（快速踢球、复杂碰撞）规划能力有限；论文展望 **可学习 contact anchor 生成**。
- **夹爪欠驱动**：当前硬件限制精细操作；contact flow 向 **灵巧手** 扩展是明确 future work。
- **仿真评测与真机条件要区分**：论文主任务成功率来自仿真 benchmark；附录 D.1 同时记载 G1 真机部署，使用外部 Noitom 动捕和 pelvis/物体 tracker 获取全局位姿（策略 50 Hz、动捕 100 Hz）。不能把这些演示结果解释成仅用机载传感器的自主部署成绩。

## 源码运行时序图

```mermaid
sequenceDiagram
    autonumber
    participant E as deploy_omnicontact/run_skill_omnicontact.py
    participant G as CFgen / NPZmotion
    participant T as CFtrack ONNX
    participant M as G1 MuJoCo 环境
    E->>G: 选择 task、goal 或 NPZ 轨迹
    E->>T: 加载 policy/omnicontact/model/omnicontact_transformer.onnx
    loop sim2sim 闭环
        M-->>G: 当前机器人与物体状态
        G->>T: Contact Flow 参考
        M-->>T: 本体观测
        T->>M: 关节控制动作
        M-->>E: 新状态与技能进度
    end
```

直接脚本和 Xbox FSM 两条入口见 [官方 sim2sim 源码归档](../../sources/repos/omnicontact-sim2sim.md)；上图描述公开 MuJoCo 执行路径，不表示真机硬件驱动代码已开放。

### 官方 CF-Track 训练源码时序（Isaac Lab）

```mermaid
sequenceDiagram
    autonumber
    participant E as scripts/rsl_rl/train.py
    participant L as Isaac Lab Gym 环境
    participant A as AMP 专家 NPZ 数据
    participant R as RSL-RL runner
    E->>E: AppLauncher 启动 Isaac Sim
    E->>L: 读取 motion_file_dir 并 gym.make(task)
    opt AMPPPO 任务
        E->>A: 解析 expert motion_files / amp_data_dir
        A-->>E: amp_data 与观测字段校验
    end
    E->>R: runner_class(env, cfg)
    E->>R: runner.learn(max_iterations)
    loop PPO 或 AMPPPO 训练
        R->>L: 策略采样与环境步进
        L-->>R: 观测、奖励与终止信息
        R->>R: 更新策略及 AMP 判别器（按算法）
    end
    R-->>E: 训练日志和 checkpoint
```

此图核对自 [官方训练仓库](https://github.com/Ingrid789/OmniContact) 的 `scripts/rsl_rl/train.py`；回放/ONNX 导出由 `scripts/rsl_rl/play.py` 实现，运行需额外配置 Isaac Lab 及动作资产，未在此处执行训练验证。

## 项目资源与工程补充

### 核心能力

| 路径 | 脚本 | 用途 |
|------|------|------|
| 脚本化 | `run_skill_omnicontact.py` | `--reference-source CFgen` 或 `NPZmotion` |
| 交互式 | `deploy_omnicontact.py` | Xbox 手柄热切换，镜像 sim2real FSM |
| CF-Track 训练 | `scripts/rsl_rl/train.py`（另一仓库） | Isaac Lab PPO/AMP、MLP/Transformer，回放见 `play.py` |

支持单 skill（`carrybox`、`pushbox`、`kickball`…）、链式 preset（`carry-push`、`carryheart`…）及 `data/` 下 NPZ 全轨迹回放。

## 关联页面

- [Loco-Manipulation](../tasks/loco-manipulation.md) — 长时程全身移动操作任务语境
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md) — 接触语义与中间表示
- [PhysHSI](./paper-amp-survey-15-physhsi.md) — AMP+HSI 单技能强基线
- [HDMI](./paper-hrl-stack-06-hdmi.md) — dense HOI co-tracking 对照
- [LessMimic](./paper-notebook-lessmimic-long-horizon-humanoid-interaction-with.md) — implicit embedding + chaining 对照
- [HumanX](./paper-hrl-stack-05-humanx.md) — 物体位姿泛化 HOI RL
- [OmniRetarget](./paper-hrl-stack-03-omniretarget.md) — 交互保留重定向数据管线
- [UMR](./paper-umr-unified-motion-retargeting.md) — 用本数据集重定向后训接触策略；相对 OmniRetarget 降关节/物体误差
- [Unitree G1](./unitree-g1.md) — 默认硬件平台
- [AMP Reward](../methods/amp-reward.md) — CF-Track 运动先验
- [Sim2Real](../concepts/sim2real.md) — sim2sim 校验入口

- [whole-body-control](../concepts/whole-body-control.md)
- [motion-retargeting](../concepts/motion-retargeting.md)
- [vla](../methods/vla.md)
- [humanoid-gym](./humanoid-gym.md)
- [visualmimic](./paper-notebook-visualmimic.md)

## 参考来源

- [sources/papers/omnicontact_arxiv_2606_26201.md](../../sources/papers/omnicontact_arxiv_2606_26201.md)
- [sources/sites/omnicontact-project.md](../../sources/sites/omnicontact-project.md)
- [sources/repos/omnicontact-sim2sim.md](../../sources/repos/omnicontact-sim2sim.md)
- [sources/repos/omnicontact-training.md](../../sources/repos/omnicontact-training.md)
- [sources/datasets/omnicontact-dataset.md](../../sources/datasets/omnicontact-dataset.md)
- Yu et al., *OmniContact: Chaining Meta-Skills via Contact Flow for Generalizable Humanoid Loco-Manipulation*, arXiv:2606.26201, 2026. <https://arxiv.org/abs/2606.26201>

- <https://github.com/Ingrid789/OmniContact_sim2sim>

## 推荐继续阅读

- [OmniContact 项目主页](https://omnicontact.github.io/)
- [GitHub: OmniContact_sim2sim](https://github.com/Ingrid789/OmniContact_sim2sim)
- [GitHub: OmniContact（Isaac Lab CF-Track 训练）](https://github.com/Ingrid789/OmniContact)
- [HuggingFace: OmniContact-Dataset](https://huggingface.co/datasets/lightcone02/OmniContact-Dataset)
- [arXiv HTML 全文](https://arxiv.org/html/2606.26201v1)
- [LessMimic（arXiv:2602.21723）](https://arxiv.org/abs/2602.21723) — 同 long-horizon chaining 议题的 embedding 路线对照（见 [LessMimic 实体](./paper-notebook-lessmimic-long-horizon-humanoid-interaction-with.md)）
