# SayCan — Do As I Can, Not As I Say: Grounding Language in Robotic Affordances

> 原始论文与项目页完整核对归档

- **标题：** Do As I Can, Not As I Say: Grounding Language in Robotic Affordances
- **类型：** paper
- **arXiv：** [2204.01691v2](https://arxiv.org/abs/2204.01691)（v1: 2022-04-04，v2: 2022-08-16）
- **作者与机构：** Michael Ahn, Anthony Brohan, Noah Brown, ..., Brian Ichter, Fei Xia, Karol Hausman 等（Robotics at Google, Everyday Robots；作者按字母序排列）
- **项目页：** <https://say-can.github.io/>
- **代码 / 数据集：**
  - 开源桌面仿真版 Colab：<https://github.com/google-research/google-research/tree/master/saycan>（**部分开源**：仅公开 UR5 桌面仿真 + CLIPort + ViLD + GPT-3 示例，未开源真机 Everyday Robots 移动操作栈、PaLM/FLAN 权重、68k 遥操作数据与 MT-Opt/BC-Z checkpoint）
  - 数据集 v0（CC BY 4.0）：<https://github.com/say-can/say-can.github.io/tree/main/data>（自然语言指令到可行技能序列映射）
- **入库日期：** 2026-11-14
- **复核日期：** 2026-11-14
- **核对范围：** `2204.01691.txt` 全文（Sec. 1–8 及 Appendix A–E.6，共 2001 行：含 Algorithm 1、Table 1–8、Listing 1–3 与 101 条评测指令全表）及项目官网 `saycan.html`（https://say-can.github.io/ 文本与链接清单）
- **通读：** 已完成
- **沉淀到 wiki：** [`wiki/methods/saycan.md`](../../wiki/methods/saycan.md)、[`wiki/entities/paper-saycan.md`](../../wiki/entities/paper-saycan.md)

## 一句话摘要

SayCan 将大语言模型的任务语义评分 $p(\ell_\pi \mid i, \text{history})$ 与底层技能的价值/可行性函数（affordance）$p(c_\pi \mid s, \ell_\pi)$ 相乘，在真实厨房移动操作机器人上把抽象长程自然语言指令落地为「既有用又物理可执行」的技能序列。

## 核心机制与算法（Sec. 3–4, Algorithm 1）

1. **问题分解（Task-grounding × World-grounding）：**
   - 给定高层指令 $i$、当前状态 $s_n$ 与技能集合 $\Pi$（各含短文本标签 $\ell_\pi \in \ell_\Pi$）。
   - **Say（LLM 打分）：** 不直接自由生成文本（可能产出「用吸尘器」等不具备的动作），而是对给定技能标签 $\ell_\pi$ 的完成似然打分 $p(\ell_\pi \mid i, \ell_{\pi_{n-1}}, \dots, \ell_{\pi_0})$。
   - **Can（Affordance 价值函数）：** 由强化学习（TD 备份、稀疏 0/1 完成奖励）学得语言条件价值函数 $p(c_\pi \mid s_n, \ell_\pi)$，度量当前状态 $s_n$ 下执行技能 $\pi$ 成功的概率。
   - **联合选择与循环：** $\pi_n = \arg\max_{\pi \in \Pi} p(c_\pi \mid s_n, \ell_\pi)\,p(\ell_\pi \mid i, \ell_{\pi_{n-1}}, \dots, \ell_{\pi_0})$；执行 $\pi_n(s_n)$ 后把 $\ell_{\pi_n}$ 追加到对话历史，循环直至选中 `"done"` / terminate。
2. **机器人与底层策略实现（Sec. 4, Appendix C & D）：**
   - 平台：Everyday Robots 移动操作机器人（7-DoF 机械臂 + 二指夹爪 + RGB 观察 + 移动底盘）。
   - 技能体系：围绕厨房场景定义 551 个技能（7 个技能族、17 个物体），真实厨房评测聚焦 15 个物体与 5 个已知语义位置。
   - **Pick：** 动作策略采用 BC-Z 架构（ResNet-18 + USE 句嵌入 FiLM 条件；68,000 条遥操作演示 + 12,000 条自主筛选成功轨迹，16 TPUv3 训练 ~27h）；Affordance 采用 MT-Opt 架构在仿真中借助 RetinaGAN sim-to-real 训练（16 TPUv3 ~100h + 3000 CPU 采集 + 3000 CPU 计算目标 Q 值），输出 Q 值经经验阈值 $v_{\min}^{\text{pick}}=0.2, v_{\max}^{\text{pick}}=0.5$ 线性归一化截断到 $[0, 1]$。
   - **Go to：** 假定已知物体位置，采用经典导航规划；affordance 由距离归一化（$d_{\min}=0, d_{\max}=100$）。
   - **Place：** 假定前置 pick 已确保语义合理性，使用笛卡尔空间经典运动规划并设 $p_{\text{place}}^{\text{affordance}}=1.0$；**Terminate** 设固定小常数 $0.1$，并在软件层抑制已完成技能的重复执行。

## 关键实验结果（Sec. 5, Table 1–8）

- **基准规模：** 7 类共 101 条指令（NL Single 15、NL Nouns 15、NL Verbs 15、Structured 15、Embodiment 11、Crowd-Sourced 15、Long-Horizon 15），在 Mock Kitchen 与 Real Office Kitchen 两处评测；指标为人工 3 评委多数决的 **Plan 成功率** 与 **Execution 成功率**。
- **主结果（Table 2, PaLM 540B）：**
  - Mock Kitchen：PaLM-SayCan **Plan 84% / Execute 74%**；显著高于 **No VF**（Plan 67%）、**Generative + USE**（Plan 74%）、**BC USE**（总 Execute 9%，仅单技能 60%、其余全 0%）与 **BC NL**（0%）。
  - Real Kitchen（真实办公厨房迁移）：PaLM-SayCan **Plan 81% / Execute 60%**（规划降 3%、执行降 14%）。
- **错误分布（Sec. 5.1, Fig. 16）：** 规划失败中 **65% 来自 LLM 错误**（长程提前终止、否定句、指代消歧等），**35% 来自 affordance 误判**。
- **LLM 规模与模型对比（Table 3 & Table 6）：**
  - PaLM-SayCan（540B，Plan 84% / Execute 74%）优于 FLAN-SayCan（137B，Plan 70% / Execute 61%），错率约减半；Generative 消融显示 PaLM 540B (74%) ≈ PaLM 62B (72%) >> PaLM 8B (38%) & FLAN 137B (43%)。
- **Prompt 示例数与扩展能力（Table 4, 5, 7, 8 & Appendix D.3/E.3–E.5）：**
  - 0-shot 无示例时带终止判定的规划成功率仅 10%，增至 4 个示例即达 82%，17 个示例（Listing 1）达 88%。
  - 抽屉扩展（21 条指令）：Plan 100% / Execute 33%（瓶颈在底层开合抽屉操作）。
  - Chain-of-Thought（加入 `Explanation:` 先生成解释再对技能打分）可解否定条件与隐式约束（如无咖啡因饮料）。
  - 多语种（中/法/西 3 组基线任务共 12 条）：无需专门微调即接近英文本体规划表现（仅法语 1 条失败）。

## 开源与局限边界（Sec. 6, Sec. 8, `saycan.html`）

- **开源版差异（Sec. 6）：** 公开的 tabletop Colab 环境采用 UR5 桌面仿真、CLIPort 抓取放置策略、**ViLD 开放词表检测器替代缺失的价值函数作 affordance**、以及 **GPT-3** 作为语言模型，输出 `"pick up X and place it in Y"`。
- **方法局限（Sec. 5.1 & Sec. 8）：**
  1. 开环假设强：仅在每步决策前读取当前 $s_n$ 的 affordance，技能中途失败但仍报高价值时缺少显式成败检测与纠错闭环（后续工作 Inner Monologue 补全）。
  2. 受限于预训练技能库与底层执行器覆盖范围；否定句、罕见指代需 CoT 与 prompt 工程缓解。
  3. 对话格式与技能词表命名（如 `"a"` vs `"an"`、step 换行与编号）对 LLM 打分敏感。

## 参考链接

- 论文全文（本地归档源）：`2204.01691.txt`（<https://arxiv.org/abs/2204.01691>）
- 项目官网（本地归档源）：`saycan.html`（<https://say-can.github.io/>）
- 开源桌面版仓库：<https://github.com/google-research/google-research/tree/master/saycan>
- 数据集 v0：<https://github.com/say-can/say-can.github.io/tree/main/data>
