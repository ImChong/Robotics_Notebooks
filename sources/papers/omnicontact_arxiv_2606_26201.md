# OmniContact: Chaining Meta-Skills via Contact Flow for Generalizable Humanoid Loco-Manipulation

> 来源归档（ingest）

- **标题：** OmniContact: Chaining Meta-Skills via Contact Flow for Generalizable Humanoid Loco-Manipulation
- **类型：** paper
- **来源：** arXiv abs / HTML；项目页、代码仓库、HuggingFace 数据集交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2606.26201>
  - <https://arxiv.org/html/2606.26201v1>
  - <https://omnicontact.github.io/>
  - <https://github.com/Ingrid789/OmniContact_sim2sim>
  - <https://huggingface.co/datasets/lightcone02/OmniContact-Dataset>
- **作者：** Runyi Yu, Xiaoyi Lin, Ji Ma, Yinhuai Wang（✉）, Koukou Luo, Jiahao Ji, Huayi Wang, Wenjia Wang, Runhan Zhang, Ping Tan, Ting Wu, Ruoli Dai, Qifeng Chen（✉）, Lei Han（✉）
- **机构：** 诺亦腾机器人（Noitom Robotics）、香港科技大学（HKUST）、武汉大学（WHU）、香港大学（HKU）
- **硬件：** Unitree G1（29 DoF）；项目页含 MuJoCo WASM + ONNX 在线 viewer 与 G1 交互 GLB 序列
- **入库日期：** 2026-07-01
- **一句话说明：** 以 **Contact Flow（CF）**——稀疏关键体轨迹 + 四端二进制接触信号——作为规划–执行共享接口：**CF-Track** 统一 RL 低层 meta-skill 库，**CF-Gen** 规则合成/在线重规划实现长时程技能链与自主恢复；配套 **OmniContact MoCap HOI 数据集**（1,274 序列 / 22.29 h）与 MuJoCo sim2sim 部署栈。

## 核心论文摘录（MVP）

### 1) 问题：长时程 loco-manipulation 需要可组合、可闭环的 skill 接口

- **链接：** <https://arxiv.org/abs/2606.26201> §I；Table 1
- **摘录要点：** 现有路线三类短板并存：**BC** 难扩展且开环慢；**body motion tracking**（SONIC 等）缺物体感知、难恢复交互失败；**dense HOI tracking**（HDMI、OmniRetarget）帧级参考难规划/编辑；**task-specific RL**（PhysHSI）单技能强但难组合；**implicit skill embedding**（LessMimic）紧凑但难解释与结构化组合。核心瓶颈是 **meta-skill 的表示与闭环链接**。
- **对 wiki 的映射：**
  - [OmniContact（论文实体）](../../wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md) — 问题定位与 Table 1 对照。
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 长时程技能链子路线。

### 2) Contact Flow：稀疏体目标 + 二进制接触时序

- **链接：** arXiv §3.2；<https://omnicontact.github.io/>
- **摘录要点：**
  - 每步 $F_t = \{(b_{t+k}, c_{t+k})\}_{k \in \mathcal{T}}$，$\mathcal{T}=\{0,1,2,3,4,8,12,16,24,32,50\}$。
  - $b_{t+k}$：稀疏 body targets（wrists、torso、ankles）；$c_{t+k} \in \{0,1\}^4$：四端接触（左/右踝、左/右腕）。
  - 相对 dense HOI 轨迹 **更可编辑/在线合成**；相对纯 object goal **保留接触时序语义**。
  - Ablation：仅 torso 跟踪成功率 **0.5%**；加 contact 位从 **11.5%→98.7%**（Carry Box）。
- **对 wiki 的映射：**
  - [OmniContact](../../wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md) — CF 定义与 ablation。
  - [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md) — 接触语义中间表示。

### 3) CF-Track：统一低层 RL 执行器

- **链接：** arXiv §3.3
- **摘录要点：**
  - 观测 $x_t = [F_t, H_t]$，$H_t$ 为 $K=5$ 步历史（本体 + 物体相对 6D pose + bbox + 上一步动作）。
  - 奖励 $r_t = \lambda_{\text{track}} r^{\text{track}} + \lambda_{\text{amp}} r^{\text{amp}} + \lambda_{\text{reg}} r^{\text{reg}}$；AMP 先验 + 跟踪 + 动作平滑。
  - **单一策略** 在 OmniContact 数据集上统一 carry / push / slide / kick / relocate 等模式；推理时跟踪 CF-Gen 启发式计划或 NPZ 全轨迹。
  - 奖励平衡 ablation：纯跟踪 $R_{\text{stable}}=46.3\%$；0.85–0.15 track–amp 达 **98.7%** 成功率。
- **对 wiki 的映射：**
  - [AMP Reward](../../wiki/methods/amp-reward.md)、[Whole-Body Control](../../wiki/concepts/whole-body-control.md)。

### 4) CF-Gen：相位模板 + 物体几何锚点 + 50 Hz 重规划

- **链接：** arXiv §3.4–3.5
- **摘录要点：**
  - 每 meta-skill 分解为 **phase blocks**（接近、预抓、抬升、搬运、释放等），库见 Appendix B.1。
  - 关键帧由 **object-centric geometry** 锚定；接触相位用约束 IK（pelvis 高度/俯仰 + 关节，排除 waist roll/yaw）求 wrist/ankle 目标。
  - 相位间 LERP/SLERP 插值得 dense 参考，再按 $\mathcal{T}$ 采样转 CF 供 CF-Track。
  - **50 Hz 闭环监控**：物体观测–预测偏差 $\delta_t > \epsilon$ 时 abort 并重规划 → 自主 recovery（掉箱重接近等）。
  - 在线重规划：Push Suitcase **82.5%→94.5%**，Stack Boxes **56.6%→80.5%**。
- **对 wiki 的映射：**
  - [OmniContact](../../wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md) — 分层 Mermaid 管线。

### 5) 仿真 benchmark 与 scaling

- **链接：** arXiv §4；Table 2–3
- **摘录要点：**
  - **Meta-skill**：Carry Box **98.7%**、Push Suitcase **82.5%**（$E_{\text{obj}}^T=0.07/0.27$ m）。
  - **Meta-skill chaining**：Stack Boxes 三阶段 **89/87/56.5%**；Push-Stack **91.5/76.5%**；基线（Sonic/HDMI/PhysHSI/LessMimic）长时程任务 **0%** 或近零。
  - 相对基线平均提升：meta-skill **+40.9%**、chaining **+66.5%**。
  - **数据 scaling**：10%→100% 数据（2.2 h→22.3 h）成功率与物体精度单调提升。
  - **耐力测试**：连续搬箱约 **40 min**（Appendix C.4）。
- **对 wiki 的映射：**
  - [PhysHSI（AMP #15）](../../wiki/entities/paper-amp-survey-15-physhsi.md)、[LessMimic](../../wiki/entities/paper-notebook-lessmimic-long-horizon-humanoid-interaction-with.md)、[HDMI](../../wiki/entities/paper-hrl-stack-06-hdmi.md) — 基线对照。

### 6) VLM 集成与语义任务

- **链接：** arXiv §3.5、§4；项目页 VLM Integration
- **摘录要点：** VLM 将自然语言分解为 **start–goal 物体位姿**；CF-Gen 用 contact anchor 模板自动合成连续 CF。示例：散箱排成心形、按颜色分拣、足球进门、拼 **N-O-I-T-O-M** 字母等。
- **对 wiki 的映射：**
  - [VLA](../../wiki/methods/vla.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 语义高层 + 接触中间层路线。

### 7) OmniContact 数据集（Appendix A + HuggingFace）

- **链接：** <https://huggingface.co/datasets/lightcone02/OmniContact-Dataset>；arXiv Appendix A
- **摘录要点：**
  - **1,274** 有效序列、**22.29 h**、**7.22M** 物体帧；90 Hz 人–物同步 MoCap（BVH + 6-DoF 刚体物体）。
  - 原语：carry / push / relocate / slide / kick；平均序列 **62.98 s**、物体路径 **19.76 m**（相对 OMOMO 5.69 s / 2.67 m 偏长时程搬运）。
  - HF 公开 **917** 条 G1 重定向 NPZ（box + soccer 子集）；键含 `joint_pos`、`contact_info`（左踝/右踝/左腕/右腕）、物体 6D pose；70/15/15 train/val/test split。
  - 项目页 t-SNE：**74,641** MoCap clips 按 skill 着色。
- **对 wiki 的映射：**
  - [OmniContact Dataset](../../sources/datasets/omnicontact-dataset.md)、[Motion Retargeting](../../wiki/concepts/motion-retargeting.md)。

## 对 wiki 的映射（汇总）

- 主实体：[OmniContact（论文实体）](../../wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md)
- 项目页：[omnicontact-project.md](../sites/omnicontact-project.md)
- 代码：[omnicontact-sim2sim.md](../repos/omnicontact-sim2sim.md)
- 数据集：[omnicontact-dataset.md](../datasets/omnicontact-dataset.md)
- 交叉：[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[sim2real](../../wiki/concepts/sim2real.md)、[unitree-g1](../../wiki/entities/unitree-g1.md)、[paper-amp-survey-15-physhsi](../../wiki/entities/paper-amp-survey-15-physhsi.md)、[paper-hrl-stack-06-hdmi](../../wiki/entities/paper-hrl-stack-06-hdmi.md)、[paper-notebook-lessmimic-long-horizon-humanoid-interaction-with](../../wiki/entities/paper-notebook-lessmimic-long-horizon-humanoid-interaction-with.md)

## 参考来源（原始）

- arXiv:2606.26201
- <https://omnicontact.github.io/>
- <https://github.com/Ingrid789/OmniContact_sim2sim>
- <https://huggingface.co/datasets/lightcone02/OmniContact-Dataset>
