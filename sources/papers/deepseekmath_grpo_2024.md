# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（GRPO 原始论文）

> 来源归档：从原先 LightNav-0 脚注补全为 DeepSeekMath 原论文核对记录；这是 **GRPO 方法原典**，并非 LightNav-0 或 AesFormer 的实验原文。

- **标题：** DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- **作者：** Zhihong Shao、Peiyi Wang、Qihao Zhu 等
- **机构：** DeepSeek-AI；清华大学、北京大学（作者署名）
- **类型：** paper / language-model / reinforcement-learning / grpo
- **原文：** <https://arxiv.org/abs/2402.03300>；[HTML v3](https://arxiv.org/html/2402.03300v3)；[PDF v3](https://arxiv.org/pdf/2402.03300v3)
- **版本：** arXiv:2402.03300v3（2024-04-27）；v1 为 2024-02-05
- **官方仓库：** <https://github.com/deepseek-ai/DeepSeek-Math>
- **入库日期：** 2026-09-21（最初建立）；**本次核对：** 2026-11-15
- **核对范围：** arXiv v3 HTML 完整正文 §1–§6（数学数据挖掘、预训练、SFT、§4 GRPO、实验及讨论、结论与局限）、Appendix A.1（各 RL/RFT/DPO/PPO/GRPO 梯度系数推导）及图表；特别复核 §4.1.1–§4.1.4、§4.2、Table 5、§5.2 与公式 (3)(4)(19)–(21)。
- **通读：** 已完成
- **提出：** GRPO（Group Relative Policy Optimization），用同题多输出组的相对分数代替单独训练的 critic 基线，降低 PPO 式 LLM 后训练的资源负担。
- **关联应用：** LightNav-0 第三阶段在线 RL（具体应用以其自身原文为准）；AesFormer 的 GRPO-A 美学动作规划是另一任务的定制变体。
- **对 wiki 的映射：** [GRPO 方法页](../../wiki/methods/grpo.md)。

## 摘录 1：提出背景与模型训练（§1–§3）

- DeepSeekMath 7B 基于 DeepSeek-Coder-Base-v1.5 7B，数学语料从 Common Crawl 迭代挖掘，约 **120B** 数学 tokens，使用多语言数学数据与代码继续预训练；经含 CoT/PoT/工具集成推理的 **776K** 例指令数据 SFT。
- 本论文同时贡献 **领域数学预训练**与 **GRPO 后训练算法**。不要将 120B tokens、776K SFT 样本与 GRPO 使用的 **约 144K** GSM8K/MATH 题目混淆。

## 摘录 2：原始 GRPO 目标与组内 advantage（§4.1.1–§4.1.2）

- 对同一道题 `q`，旧策略 `π_old` 采样 `G` 个回答 `o_i`；使用奖励模型打分 `r_i`。
- **结果监督（OS）**下，组内优势是 `A_i,t = (r_i - mean(r_1,…,r_G)) / std(r_1,…,r_G)`，同一回答所有 token 共享该值；因此**不是仅简单把回答排序**，而是原文明确的分组标准化分数。
- 目标沿用 PPO 式 token 重要性比率与 clipped surrogate，另在目标中直接加入相对参考策略的 **KL 正则**，避免把 KL penalty 混进用于计算 advantage 的奖励。
- 原文 Fig.4/§4.1 强调：不再单独训练与 policy 规模相近的 value/critic 模型，改用**同组回答分数作 baseline**，可减少额外显存/计算负担；但并非无需奖励模型或推理采样。

## 摘录 3：过程监督与迭代 RL（§4.1.3–§4.1.4）

- **过程监督（PS）**以每个推理步骤末端的奖励为信号，将跨样本的步骤奖励归一化；一个 token 的 advantage 是其之后（含当前步结束位置）各步归一化奖励之和：`A_i,t = Σ_{index(j)≥t} r̃_i^{index(j)}`。
- Iterative GRPO 会根据最新 policy 的样本更新奖励模型，replay 引入 **10%** 历史数据；外层迭代重置 reference policy。不能把这一机制误写成普通单轮在线 GRPO 的必备部分。

## 摘录 4：实验设置与结果（§4.2、Table 5）

| 项 | 论文设置 / 结果 |
|---|---|
| GRPO 题目 | SFT 数据的约 **144K** 道 GSM8K + MATH CoT 题 |
| 采样 | **64** 个回答/题；max length 1024；batch size 1024 |
| GRPO | 策略学习率 **1e-6**；KL 系数 **0.04**；采样后仅一次策略更新 |
| GSM8K（CoT，Top1） | Instruct **82.9% → RL 88.2%** |
| MATH（CoT，Top1） | Instruct **46.8% → RL 51.7%** |
| CMATH（CoT，Top1） | Instruct **84.6% → RL 88.8%**（RL 未使用该基准题作在线训练） |

- §5.2 对比显示：在线采样优于离线 RFT；GRPO 通过正负梯度系数区别于只强化正确样本的 Online RFT；过程监督优于仅结果监督的设定。结果依赖具体数学任务和奖励模型，不宜推导出所有机器人任务的同幅收益。

## 摘录 5：局限与可迁移的读法（§5–§6、Appendix A.1）

- 作者观察 RL 提升 **Maj@K**，但没有显著提升 **Pass@K**，认为它更可能提高已存在正确答案的概率质量，而未证明扩大模型可解问题的集合。
- DeepSeekMath 对几何和定理证明、few-shot 仍弱于更强闭源模型；单组内标准化优势可能在同组奖励接近时提供弱区分信号（这属于机制上的工程推论，不是论文给出的一般定理）。
- 论文统一比较 SFT、RFT、Online RFT、DPO、PPO、GRPO 的 **数据来源 / 奖励来源 / 梯度系数**；不同领域迁移时必须重新设计 reward 与采样预算。
- 在 AesFormer，GRPO-A 迁移的是组内相对策略优化思路，奖励则换成 **格式 / 摄影编辑意图一致性 / 创造性**，与 DeepSeekMath 的数学推理 reward 不同；不要混同原论文的结果数字。

## 一手链接

- [DeepSeekMath arXiv v3 全文 HTML](https://arxiv.org/html/2402.03300v3)
- [DeepSeekMath PDF](https://arxiv.org/pdf/2402.03300v3)
- [DeepSeek-Math 官方 GitHub](https://github.com/deepseek-ai/DeepSeek-Math)
