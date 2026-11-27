# Reflective Planning: Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation

- **作者：** Yunhai Feng, Jiaming Han, Zhuoran Yang, Xiangyu Yue, Sergey Levine, Jianlan Luo
- **机构：** Cornell University、香港中文大学（CUHK）、Yale University、UC Berkeley
- **版本 / 载体：** 9th Conference on Robot Learning (CoRL 2025, PMLR v305, Seoul, Korea)；官方开源仓库标注 arXiv:2502.16707
- **原文：** https://proceedings.mlr.press/v305/feng25b.html （PDF：https://raw.githubusercontent.com/mlresearch/v305/main/assets/feng25b/feng25b.pdf ；arXiv 入口：https://arxiv.org/abs/2502.16707 ）
- **项目页 / 代码：** https://github.com/yunhaif/reflect-vlm （项目主页：https://reflect-vlm.github.io/ 及 https://corl2025-reflectvlm.github.io ；**部分开源（推理/仿真/权重已开源，后训练脚本待补充）**：MIT 协议公开 MuJoCo 仿真任务环境、评测脚本 `eval_base_vlm.sh` / `eval_reflect_vlm.sh`、扩散演示脚本及 Hugging Face 预训练/后训练 VLM 与扩散动力学模型权重；仓库 README 中「Policy training」标注为 Coming soon）
- **核对范围：** 通读论文全文（Abstract、Sections 1–7、References 与 Appendix A–J，含 Figures 1–16 与 Tables 1–8，共 25 页正文与附录），并核对官方 GitHub 仓库 `yunhaif/reflect-vlm` 主页结构与 README 入口。
- **通读：** 已完成

## 原文核对笔记

### 1. 问题定义与研究动机

- **多阶段长程物理约束操作（Section 1, 3, 5）：** 论文聚焦多块互锁积木在桌面上的多阶段装配任务。不同积木之间存在强拓扑依赖与几何遮挡关系（部分积木需按严格顺序插入，或初始即处于阻挡位置，需要先移开阻挡件、重新规划后再装回）。
- **POMDP 建模（Section 3）：** 将任务建模为部分可观测马尔可夫决策过程 $(S, A, T, O, Z)$：
  - 状态 $S$ 包含所有物体位姿与物理属性；策略仅能访问当前 RGB 图像 $I_t$ 与目标 RGB 图像 $I_g$（外加最近 $h=10$ 步历史动作文本）。
  - 高层动作空间为 $A = \{\text{pick up}, \text{insert}, \text{reorient}, \text{put down}\} \times \{\text{objects}\}$，每个规则脚本原语带有真实执行失败概率 $\epsilon$（Table 5 统计为 90.1%–98.4%）。
  - 训练阶段假设存在可读取真值状态的符号专家策略 $\pi_E(s_g, s_t)$（在评测集上成功率为 97%）。
- **现有方案瓶颈：** 直接使用闭源商业 VLM 做零样本规划缺乏物理交互常识与长程防错能力；基于专家状态价值的 MCTS（50 次迭代）不仅每步耗时高达 391.42 s，而且对价值函数估计极为敏感，容易错误覆盖基础微调 VLM 原本正确的动作计划（成功率从 47.8% 跌至 24.0%）。

### 2. ReflectVLM 核心方法（Section 4 & Appendix H, I）

- **交互式 VLM 后训练（Section 4.1, Algorithm 1, Fig. 2）：**
  1. 先用 5,000 条专家演示（1,000 个独特任务 × 5 种随机初始桌面布局）对 `LLaVA-1.5-13B`（视觉编码器 `clip-vit-large-patch14-336` + 语言底座 `vicuna-13b-v1.5`，LoRA rank 128）做行为克隆（BC）预训练。
  2. 在 $K=3$ 轮 DAgger 式交互后训练中，每轮从 1,000 个训练任务中采样 200 个，滚动采集 1,000 条轨迹（按概率 $p$ 执行学习者动作 $a^\dagger_t \sim \pi_{\text{VLM}}(I_g, I_t)$，以 $1-p$ 执行专家动作 $a^*_t$）。
  3. 对每条轨迹在结束后做**反思数据重标（Relabeling）**：同一时间步 $t$ 同时构造两类问答样本——动作提议 $(Q_1, A_1)$（输入 $I_g, I_t, a_{t-h:t-1}$ 监督 $a^*_t$）与反思修正 $(Q_2, A_2)$（额外引入执行 $H=5$ 步动作序列后的未来观测 $I_{t+H}$ 及候选计划文本，仍监督最优动作 $a^*_t$）。单一 VLM 通过两套提示词模板联合优化交叉熵损失（Eq. 1）。
- **扩散动力学模型 DDM（Section 4.2, Fig. 3, Appendix H.2）：**
  - 基于预训练图像编辑模型 `timbrooks/instruct-pix2pix` 微调单步视觉前瞻模型 $\tilde{T}(I_t, a_t) \to \tilde{I}_{t+1}$。
  - 冻结 Latent Encoder 与 Text Encoder，分别并行训练 Diffusion UNet（512 px，20K 步，batch 640，LR `1e-4`）与 Latent VAE Decoder（512 px，4K 步，batch 160，LR `1e-7`，专门提升桌面上小尺寸积木的重建保真度）。
  - 训练数据为 10,000 个装配板通过带随机噪声（随机动作替换概率 $p \in \{0.2, 0.5, 0.7, 0.9, 1.0\}$）的次优策略采集的 50,000 条轨迹（约 100 万条转移，其中 50K 用于验证；文中第 4.2 节说明也加入了少量测试数据点以提高 DDM 保真度）。
- **测试时反思规划（Section 4.3, Algorithm 2）：**
  - 在真实环境每步决策前，VLM 先用提议模板结合 DDM 自回归前瞻推演 $H=5$ 步，得到想象未来图像 $\tilde{I}_{t+H}$ 与草拟序列 $\tilde{a}_{t:t+H-1}$；随后将 $(I_t, I_g, \tilde{I}_{t+H}, \tilde{a}_{t:t+H-1})$ 送入同一 VLM 的反思模板，判别并修正首步要执行的动作 $a_t$。

### 3. 实验结果与关键消融（Section 6, Tables 1–3, Appendix A, J）

- **100 个未见多阶段装配任务（5–8 个零件，专家需 4–22 步、均值 12.13 步）：**
  - **零样本商用/开源 VLM（单次运行，Table 8 & Fig. 5）：** LLaVA-OneVision 0.0%（无法稳定遵循输出格式）、Gemini-2.0-flash 6.0%、GPT-4o 6.0%、Gemini-2.0-flash-thinking 8.0%、GPT-o1 15.0%。
  - **规划与模仿基线：** 结合预训练 VLM 的 MCTS 为 **24.0%**（单步推理耗时 **391.42 s**）；纯专家数据集监督微调基线（BC）为 **47.8%**（5 种子均值）。
  - **ReflectVLM 消融（5 种子均值，Iter. 3）：**
    - `w/o reflect`（无反思训练与推理的交互微调）：**77.8%**
    - `w/o reflect@test`（训练期含反思提示词、测试期不做前瞻反思）：**82.2%**（单步仅 **0.45 s**）
    - `reflect w/ diffusion`（完整 ReflectVLM + DDM 前瞻）：**82.4%**（单步 **11.10 s**）
    - `reflect w/ sim`（测试期用真实仿真器前瞻的上限）：**85.4%**（单步 **6.05 s**）
- **按难度分层收益（Table 3，5 种子均值）：**
  - 简单任务（24 题，0–9 步）：`w/o reflect` 83.3% → `w/o reflect@test` 85.8% → `reflect w/ diffusion` **93.3%** → `reflect w/ sim` **97.5%**。
  - 中等任务（60 题，10–14 步）：`w/o reflect` 86.3%、`w/o reflect@test` 88.7%、`reflect w/ diffusion` 82.7%、`reflect w/ sim` 85.0%。
  - **高难度长程任务（16 题，$\ge 15$ 步，最高 22 步）：** `w/o reflect` **37.5%** → `w/o reflect@test` **52.5%** → `reflect w/ diffusion` **65.0%** → `reflect w/ sim` **68.8%**，验证了反思机制对需要拆解重排与多轮重规划的复杂长程任务提升最大。

### 4. 局限性与知识映射

- **局限性（Section 7）：** 反思阶段受限于 VLM 上下文长度仅输入终末想象图像 $\tilde{I}_{t+H}$，缺乏中间步细粒度信用分配；测试时仅执行单轮反思；DDM 在更长步数下存在预测误差累积（也是 82.4% 略低于仿真器上界 85.4% 的原因）；实验仅在 MuJoCo 仿真环境验证，尚未在真实机械臂闭环部署；后训练依赖符号专家策略且官方仓库暂未公开后训练脚本。
- **对 wiki 的映射：** [ReflectVLM：面向多阶段长程机器人操作的视觉-语言模型反思规划](../../wiki/entities/paper-agentic-survey-pmlr-v305-feng25b.md)
