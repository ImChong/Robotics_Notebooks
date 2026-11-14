# Prism-GRPO（arXiv:2608.17423）

> 来源归档（ingest）

- **标题：** Prism-GRPO: Faster VLA Policy Optimization via Splitting Same-outcome Groups
- **类型：** paper / vla / reinforcement-learning / grpo / sample-efficiency / robotwin
- **arXiv：** <https://arxiv.org/abs/2608.17423>（v1，2026-08-18）
- **作者：** Zeyun Deng（Purdue University，AWS AI 暑期实习）、Yuzhe Lu、Yawei Wang、Linbo Liu、Qing Ping、Han Ding、Guande Wu、Panpan Xu、Jun Huan（AWS AI）
- **致谢：** 真机平台与相机同步支持来自 UCLA Mobility Lab（Piper 机械臂），实习学术支持致谢 Purdue DT Lab（Ziran Wang、Ruqi Zhang）
- **项目页 / 代码：** 无（arXiv 原文正文中未提供独立项目页或 Prism-GRPO 专用代码仓 URL；仅依赖公开基座 [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL) 与 OpenVLA-OFT SFT checkpoint）
- **入库日期：** 2026-08-20
- **核对日期：** 2026-11-14
- **通读：** 已完成
- **核对范围：** 全文通读 `2608.17423.txt`（Abstract、§1–§6、Eq. 1–26、Algorithm 1–2、Table 1–2、Appendix A–I）及 arXiv HTML（`2608.17423.html`），核对作者姓名断句、RLOO 与 GRPO 组内方差归一化消融、6 项轨迹质量信号归一化参数、RoboTwin 2.0（`22%–56%` rollout 节省，Table 1 `λ` 消融）与真机 Piper 25-trial 结果，并确认原文不含独立代码仓或项目页 URL。
- **一句话说明：** 在 SimpleVLA-RL 二值成功 GRPO 上叠加有界任务无关轨迹质量 `R_combined(τ) = success(τ) + λ q(τ)`（`0 < λ < 1`，默认 `λ=0.2`）并采用 RLOO 优势与自适应 gap-fill 重采样，打破 all-success / all-failure 同结果组退化，保持 success 严格支配的同时将达峰 rollout 消耗降低 `22%–56%` 并抑制真机 shove-cheat。

## 开源状态（步骤 2.5）

- **独立代码与项目页：** 无。在 `2608.17423.txt` 与 `2608.17423.html` 中均未发现任何 Prism-GRPO 官方 GitHub 仓库或项目页链接；截至核对日（2026-11-14），Prism-GRPO 补丁本身**未开源**。
- **公开基座依赖：** 论文 §5 与 Appendix F 明确基于公开的 [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL) 训练框架与 OpenVLA-OFT 任务 SFT checkpoint，在仿真环境（RoboTwin 2.0）与真机（Piper）上复用其 RL 训练流并替换奖励、优势估计（RLOO）与自适应 gap-fill 批补齐逻辑。
- **结论：** **基座可复现、Prism 算法补丁未开源** —— 复现需自行在 SimpleVLA-RL 之上按论文 Algorithm 1–2 与 Appendix B 实现质量信号提取、合并奖励、`RLOO` 优势以及自适应批补齐。

## 摘录 1：问题定义与核心方法（§1、§3–§4、Algorithm 1–2）

- **退化组浪费：** Binary GRPO（如 SimpleVLA-RL）在同一初始场景 `x` 采样 `G` 条轨迹 `τ_1,...,τ_G`（默认 `G=8`）；若全成功或全失败，组内奖励恒等导致优势 `A_i = 0`、策略梯度为零，dynamic sampling 必须丢弃该组并反复重采样。训练早期任务成功率低（几乎全失败），Lift Pot 等任务早期组丢弃率高达 `70%`。
- **结合奖励（Eq. 1）：**
  $$R_{\mathrm{combined}}(\tau) = \mathrm{success}(\tau) + \lambda q(\tau), \qquad 0 < \lambda < 1,\quad q(\tau)\in[0,1]$$
  失败轨迹落入 `[0, λ]`，成功轨迹落入 `[1, 1 + λ]`。只要 `0 < λ < 1`，任意成功轨迹始终严格优于任意失败轨迹（success dominance），同时在同结果组内依据 `q(τ)` 产生非零排序差异。
- **为何必须用 RLOO 而非标准 GRPO 组归一化（Eq. 22–23、Appendix G.6）：**
  $$A_i = R_{\mathrm{combined}}(\tau_i) - \frac{1}{G-1}\sum_{j\neq i} R_{\mathrm{combined}}(\tau_j)$$
  - 在同结果组（`success(τ_j) ≡ b`）中，常数 `b` 相减抵消，仅剩 `A_i = λ (q_i - mean_{j≠i} q_j)`，精确保留超参 `λ` 对梯度幅度的约束；
  - 若使用传统 GRPO 除以组内标准差，在同结果组内 `std` 与 `λ` 同比例缩放并被约去，等效放大质量梯度并在混合组/纯质量组间造成更新幅度剧烈波动（Appendix G.6 Fig. 22 实测会破坏成功率收敛）。
- **自适应 gap-fill 采样（Algorithm 2）：** SimpleVLA-RL 每次不足目标批大小 `B`（512 kept rollouts，即 64 个场景 × `G=8`）时直接整批再跑 512 条轨迹导致尾盘大量溢出丢弃；Prism-GRPO 用保留率估计 `r_keep = max(0.3, N_keep / N_gen)`，按 `ceil(1.2 * B_short / r_keep)` 向上取数据并行卡数整数倍并封顶 `B=64` 补齐。
- **理论保证（§4.2–§4.3）：**
  - **Theorem 1：** 对任意成功率 `p ∈ (0,1)`，构造信息组期望消耗 `C_combined(p) = G / (1 - δ(p)) ≤ C_binary(p) = G / (1 - p^G - (1-p)^G)`，且在 `p→0` 或 `p→1` 时仍然有界（Binary GRPO 趋于无穷）。
  - **Proposition 1：** 单步 softmax 策略下，若局部相关性 `ρ(s) ≥ (κ_π - 1)/(κ_π + 1)`（`κ_π = max_a π(a|s)/min_a π(a|s)`），则 `⟨g_success(s), g_quality(s)⟩ ≥ 0`，即质量梯度不损伤成功期望梯度。
  - **Proposition 2：** 对齐成立时 `⟨∇J, g_combined⟩ ≥ ||g_success||²` 且 `⟨∇Q, g_combined⟩ ≥ λ||g_quality||²`，联合更新同时推动成功率与执行质量上升。
  - **Eq. 2 协方差分解：** 总相关性分解为同场景内 `E_x[Cov(q, success | x)]` 与跨场景 `Cov_x(q̄(x), s̄(x))`，前者决定同结果失败组内质量是否指向成功高概率区域。

## 摘录 2：六类任务无关质量信号与标定（Appendix B.1–B.2、Eq. 17–21）

统一截断线性归一化：基于 256 个 SFT 验证场景轨迹标定基线阈值 `(r_0, T)`，RL 前固定不变：
$$q(\tau) = \max\left(0,\; 1 - \frac{\max(0,\, r(\tau) - r_0)}{T}\right) \in [0,1]$$

| 信号名 | 类别 | 原始量测与配置（Appendix B.1–B.2） | Lift Pot 对 Binary 节省（§5.2） |
|---|---|---|---|
| **Prism-Peak**（默认 GT-Max Force） | 非目标接触 | 机械臂/目标外非预期接触脉冲峰值；`r_0=0`，`T_peak = 10 N·s`（Lift Pot）/ `6`（Move Can Pot）/ `3`（Handover Block、Beat Block Hammer） | **56%** |
| **Prism-Count** | 非目标接触 | 非预期接触总次数；`r_0=0`，`T_n = 30` | `38–56%` 区间内 |
| **Prism-VLM-Contact** | 视觉零样本接触 | `Qwen3-VL-235B` 均匀抽 8 帧判碰撞概率；256 条轨迹判别准确率 `78.9%`（Contact F1 `69.0%`、Precision `90.9%`、Recall `55.6%`）；耗时占单步 wall-clock `4.8%`（对比仿真器接触统计仅 `0.04%`） | **38%** |
| **Prism-Flips** | 动作平滑度 | 12 维 ALOHA 手臂关节（排除 2 个夹爪）速度反向总次数；`r_0=0`（Lift Pot、Handover Block）或 `12`（SFT 中位基线：Move Can Pot、Beat Block Hammer），`T_flips = 12` | `25–44%` 区间内 |
| **Prism-MeanFlips** | 动作平滑度 | 按轨迹长 `L_τ` 归一化的关节反向次数；`T_meanflips = 7` | `25–44%` 区间内 |
| **Prism-Jerk** | 动作平滑度 | 轨迹三阶差分（jerk）RMS；`T_J = 10` | `25–44%` 区间内 |

## 摘录 3：实验、消融与真机验证（§5、Tables 1–2、Appendices C/F/G/H/I）

- **仿真配置（Appendix F）：** **RoboTwin 2.0** 四任务（Lift Pot、Move Can Pot、Handover Block、Beat Block Hammer）；OpenVLA-OFT 离散 256-bin 动作头（chunk 25×14）；`G=8`、`512` kept rollouts/step（64 场景）；`lr=5e-6`、PPO clip `(0.2, 0.28)`、采样温度 `1.6`、无 KL/熵惩罚；5 seeds × 8×H100-80GB（单次约 12–16 h，单 rollout ~0.81 s）。
- **主效率结果（Fig. 3）：** 在达到 Binary GRPO 峰值平均成功率处，四任务总 rollout 节省 **`22%–56%`**：**Lift Pot 节省 `56%`（`20.5K` rollouts）**，早前期退化组丢弃率从 Binary GRPO 的 `70%` 降至初期近 `0%`、中后期约 `14%`（Binary 约 `20%`）；**Move Can Pot 节省 `41%`（`14.6K` rollouts）**；其余两任务（Handover Block、Beat Block Hammer）节省分别为 `22%–38%` 区间内。
- **超参 `λ` 消融（Table 1，`CS` = 计算节省%，`Q` = 成功样本质量提升%）：**
  - Lift Pot (`λ=0.2/0.5/0.9/1/2`)：`CS = 56% / 49% / 45% / 25% / 53%`，`Q = +11% / +14% / +15% / +14% / +26%`；Random tie-break 在任意 `λ` 下均失败（`×`）。
  - Move Can Pot (`λ=0.2/0.5/0.9/1/2`)：`CS = 41% / 48% / 41% / 3% / ×`，`Q = +7% / +10% / +13% / +16% / ×` — 当 `λ ≥ 1` 破坏 success 支配条件时，高 `λ` 导致大幅退化甚至训崩。
- **仿真与真机抑制作弊（Fig. 5、Table 2、Appendix I）：**
  - Move Can Pot 仿真 shove-cheat 率（推动锅体 `≥10 cm` 取巧过关）：RL-ZVP `0.6–20.0%`（均值 `5.3%`）、Binary GRPO `0.7–7.0%`（`3.5%`）、**Prism-GRPO `0.6–1.3%`（`0.7%`）**。
  - **Piper 真机评测（Table 2，每方法 25 trials，Orbbec DaBai DC1 640×480→240×320，7D 动作带 `η=0.05 rad` 步长限幅，从约 65% 仿真 checkpoint 评测）：**
    - **Clean Success：** Binary GRPO `4/25 (16%)`，RL-ZVP `2/25 (8%)`，**Prism-GRPO `6/25 (24%)`**；
    - **Shove-Cheat：** Binary GRPO `1/25 (4%)`，RL-ZVP `5/25 (20%)`，**Prism-GRPO `0/25 (0%)`**。
- **关键结构消融（Appendix G & H）：**
  - **组大小 `G`（App. G.4）：** 增益在 `G=8` 最大、`G=16` 缩小、`G=32` 与 Binary 几乎无异（因 `p^G + (1-p)^G` 已急剧减小）；
  - **全组 vs 仅同结果组（App. G.3）：** 将质量项应用于**全部组**比仅用于同结果组或仅用于全失败组效果提升近一倍；
  - **`λ` 衰减（App. G.5）：** 线性/分段衰减无稳定正收益；
  - **对齐诊断（App. H，每任务 256×56=14,336 条 SFT 轨迹）：** 四任务同场景内质量–成功均值差 `0.55 / 0.45 / 0.87 / 0.85`（`89.5%–100%` 场景呈正差），跨场景 Spearman 相关系数 `0.79 / 0.68 / 0.98 / 0.80`。

## 对 wiki 的映射

- 沉淀与核对 [`wiki/entities/paper-prism-grpo.md`](../../wiki/entities/paper-prism-grpo.md)，补充精确消融表（Table 1 & 2）、RLOO 与 GRPO 归一化差异以及无官方代码仓核查结论。
- 对照页：[`Temporal GRPO`](../../wiki/entities/paper-temporal-grpo.md)（阶段信用分配 vs 同结果组质量破平）、[`VLA 开源复现景观`](../../wiki/overview/vla-open-source-repro-landscape-2025.md)（SimpleVLA-RL 生态）、[`RoboTwin`](../../wiki/entities/robotwin.md)。
