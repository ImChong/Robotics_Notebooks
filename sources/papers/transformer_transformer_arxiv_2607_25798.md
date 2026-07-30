# Transformer Transformer: A Unified Model for Motion-Conditioned Robot Co-design

> 来源归档（ingest）

- **标题：** Transformer Transformer: A Unified Model for Motion-Conditioned Robot Co-design
- **类型：** paper（运动条件机器人共设计 / 跨具身控制 / 扩散 Transformer）
- **状态：** arXiv preprint（2026-07）
- **原始链接：**
  - 项目页：<https://transformer-transformer.github.io/>
  - arXiv abs：<https://arxiv.org/abs/2607.25798>
  - PDF：<https://arxiv.org/pdf/2607.25798> · 项目页镜像 <https://transformer-transformer.github.io/static/paper.pdf>
  - 代码：<https://github.com/real-stanford/transformer-transformer>
  - 权重/数据（小包）：<https://real.stanford.edu/transformer-transformer/>
  - 训练 Zarr（HF）：<https://huggingface.co/datasets/hqhuy/transformer-transformer>
- **作者：** Huy Ha, C. Karen Liu, Shuran Song
- **机构：** Stanford University；Columbia University
- **入库日期：** 2026-07-30
- **一句话说明：** 用 **RoboTokens** 统一编码刚体关节机器人的 embodiment / state / action，训练单一 **DiT（Transformer Transformer）**，同一权重既可做 **motion→完整机体生成**，也可做 **跨具身跟踪控制**；推理时用自有动力学预测把任意可微奖励转成引导信号（**Dynamics Self-Guidance**），相对 CMA-ES 在三设计空间上更快、更高回报，并真机制造优化后的 ALOHA 抛布展开（跟踪误差 −73%、峰值关节速度 −30%）。

## 核心论文摘录（MVP）

### 1) 问题：motion-conditioned robot co-design

- **链接：** arXiv §1；项目页「What is the best robot…」
- **摘录要点：** 操作性能常被 embodiment 上限卡住。任务以 **末端轨迹**（UMI 人类示范）表示；目标是在用户定义奖励下，自动生成 **完整刚体关节机器人**（连杆几何、运动学、惯量、电机）并给出可验证的控制器。奖励可同时依赖机体属性（尺寸/质量）与跟踪行为（跟踪误差、力矩、速度）。命名双关：第一个 Transformer = 可变形态机器人；第二个 = self-attention。
- **对 wiki 的映射：**
  - [Transformer Transformer 实体页](../../wiki/entities/paper-transformer-transformer.md) — 问题定义与 Demonstrate→Generate→Validate 闭环。

### 2) RoboTokens：统一机器人表征

- **链接：** §2.1；项目页 PART 1
- **摘录要点：**
  - 五类 embodiment token（link / fixed joint / revolute·prismatic / ball / motor）+ 状态 token + action token；关节用 link ID、电机用 joint ID 互指，支持被动关节与可变 DoF。
  - 相对 MJCF 文本：**27–110× 更紧凑**（Menagerie 11 机 → 序列长约 28–101）；连续值可扩散，具备全局可控与可微性（便于推理期奖励梯度）。
  - 预处理强制变换约定、折叠冗余空间偏置，降低等价表述方差。
  - 可扩展：为轨迹跟踪追加 **target EE pose** 条件 token（不噪声）。
  - 范围：刚体关节 + **primitive 几何**；场景/接触目标/任意 mesh/可变形尚未纳入。
- **对 wiki 的映射：**
  - 同上 — 「核心原理 / RoboTokens」。

### 3) 统一架构：掩码条件切换角色

- **链接：** §2.2；Fig. 4
- **摘录要点：** DiT + DDIM；对不同 token 加噪/条件化即可切换：
  - **Motion-to-robot / hardware_gen：** 条件于目标轨迹，联合扩散 embodiment + dynamics；Zeroth-Order = 并行采样 $n$ 个候选，用模型预测动力学上的奖励打分取最优。
  - **Cross-embodiment control：** 条件于 embodiment + 状态 + 目标运动，扩散/预测动作；embodiment-aware，可跨离散/连续形态变化跟踪。
  - 训练：token-type 投影 + ID embedding；平面 SE(2) 增广；子采样 **8 timesteps**/样本；固定基座/轮式双臂用 **Mink DiffIK**，腿式用 **每离散选择一条 RL expert**（连续参数进观测；四足空间共 128 专家）。
- **对 wiki 的映射：**
  - 同上 — 流程总览与工程实践表。

### 4) Dynamics Self-Guidance：零样本奖励优化

- **链接：** §2.3；项目页 PART 3
- **摘录要点：** 模型预测整段 episode 的 state/action；把用户可微奖励接到预测上，按 classifier-guided DDIM 把 $\nabla_{\mathrm{embodiment}}$ 注入每步去噪。相对「另训动力学模型」或「可微仿真引导」：引导信号来自 **同一网络** 的动力学头。多轨迹任务用 **diffusion composition** 平均噪声预测。Guidance 与 Zeroth-Order 在采样预算大时接近；单样本/昂贵采样时 guidance 优势更大。
- **对 wiki 的映射：**
  - 同上 — 方法与结论要点。

### 5) 实验与真机

- **链接：** §3；Fig. 5–10
- **摘录要点：**
  - 三设计空间：ViperX 固定基（运动学）、UMI-on-Legs 四足操作（动力学/WBC）、轮式双臂洗碗（任务复杂度/双臂协调）。
  - vs Random / CMA-ES：GPU 并行候选 × 非自回归整段动力学评估 → 秒级～分钟级达到/超过 CMA-ES（多轨迹双臂 CMA-ES >3h，本文 <1min）。
  - 扩散 embodiment 落在训练流形内（不外推拓扑/长度范围）；奖励/轨迹改变会重塑离散+连续设计分布。
  - 自验证控制：与 RL oracle 奖励 Pearson **r≈0.53**。
  - **ALOHA2 真机抛布：** Tracking Velocity 奖励优化 → 更长连杆 + 倒置安装/下摆；跟踪误差 13.0→3.5 cm（−73%），峰值关节速度 2.57→1.82 rad/s（−30%）。
- **对 wiki 的映射：**
  - 同上 — 评测表、结论、与 ALOHA / Shape Your Body 对照。

### 6) 开源与复现入口（步骤 2.5）

- **链接：** 项目页 Code；GitHub README / `docs/starter.md`
- **摘录要点：** **已开源**（MIT，部分 DiT/MAE 文件保留上游非商用许可）。仓库含 RoboTokens、程序化设计空间、Mink/RL 数据生成、训练（`scripts/train.py`）、推理评测（`evaluate_ctrl.py` / `evaluate_hardware_opt.py`）、CMA-ES 基线与 Blender 可视化。预训练 ckpt + 评测轨迹：`real.stanford.edu/transformer-transformer`；大规模训练 Zarr：HF `hqhuy/transformer-transformer`。
- **对 wiki 的映射：**
  - [`sources/repos/transformer-transformer.md`](../repos/transformer-transformer.md)、实体页「源码运行时序图」。

## 相关资料（交叉核对）

| 资料 | 链接 | 备注 |
|------|------|------|
| 项目页归档 | [transformer-transformer-github-io.md](../sites/transformer-transformer-github-io.md) | Demonstrate/Generate/Validate、真机对比、局限 |
| 代码归档 | [transformer-transformer.md](../repos/transformer-transformer.md) | 入口脚本、ckpt、HF 数据 |
| 共设计对照 | [Shape Your Body](shape_your_body_arxiv_2606_00702.md) | VGDS：价值梯度搜连续参数；本文：扩散生成完整机体+控制 |
| 硬件语境 | [ALOHA wiki](../../wiki/entities/aloha.md) | 真机抛布平台 |
| UMI | 项目引用 Chi et al. RSS 2024 | 示范轨迹来源 |

## 当前提炼状态

- [x] 项目页 + arXiv HTML/摘要方法与结果已摘录
- [x] 开源核查：GitHub 全栈 + 官方 ckpt/数据（2026-07-30）
- [x] wiki 映射：`wiki/entities/paper-transformer-transformer.md`
