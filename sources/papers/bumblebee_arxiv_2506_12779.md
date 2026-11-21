# From Experts to a Generalist: Toward General Whole-Body Control for Humanoid Robots (BumbleBee)

> 来源归档（paper / arXiv:2506.12779v3，NeurIPS 2025 Spotlight）

- **URL：** <https://arxiv.org/abs/2506.12779>
- **PDF：** <https://arxiv.org/pdf/2506.12779.pdf>
- **项目页：** <https://beingbeyond.github.io/BumbleBee/>
- **代码仓库：** <https://github.com/BeingBeyond/BumbleBee>
- **标题：** From Experts to a Generalist: Toward General Whole-Body Control for Humanoid Robots
- **作者：** Yuxuan Wang, Ming Yang, Ziluo Ding, Yu Zhang, Weishuai Zeng, Xinrun Xu, Haobin Jiang, Zongqing Lu
- **机构：** BeingBeyond（智在无界）、Peking University、University of Illinois Urbana-Champaign、Shanghai Jiao Tong University、Sichuan University
- **年份：** 2025（arXiv v3；NeurIPS 2025 Spotlight）
- **核对范围：** 已获取并通读 `arXiv:2506.12779v3` PDF 抽取全文（Sections 1–5 及 Appendix，涵盖 AE+BERT 聚类、Base PPO 与 K=6 簇专家微调、2 轮真机 ASAP delta-action 标定及 Gated Transformer-XL DAgger 蒸馏）；未独立复现训练与真机测试。
- **通读：** 已完成

## 摘要与核心贡献

1. **Expert-to-Generalist 全身控制框架（BumbleBee, BB）**：针对高多样人类动作在同一通用 WBC 策略下产生梯度冲突（如跳跃与缓慢蹲起），先对 8,179 条 PHC 清洗后的 AMASS 轨迹提取运动 Autoencoder 与 BERT 文本语义嵌入（联合优化 InfoNCE + L2 + Huber 损失），经 K-means 聚为 **K=6 行为簇**（`Jump`、`Walk-slow`、`Walk-fast`、`Stand-up`、`Stand-mid`、`Stand-low`）。
2. **阶段化专家训练与真机残差对齐**：
   - 全量数据预训练 **Base PPO** 策略后，在各簇内分别微调为 6 个专业 **Expert**；
   - 对真机 Unitree G1（29 DoF，其中手腕等 6 个未激活，实际控制 **23 active DoF**）引入 2 轮真机数据收集 + **ASAP 风格残差动作（delta action）动力学模型**（聚焦脚踝等 4 维核心关节，2 层隐藏维 64 MLP）在仿真中微调缓解 Sim2Real 动力学失配；
   - 最后通过 **DAgger KL 散度蒸馏**将 6 个微调后的专家整合进单个带 **Gated Transformer-XL** 记忆骨干的通用全身策略。
3. **关键定量与真机结果**：
   - MuJoCo 跨仿真器迁移成功率达 **66.84%**（显著高于 ExBody2 `50.19%`、Hover `16.12%`、OmniH2O `15.64%`）；
   - 真机迭代微调（Iter 0 -> Iter 1 -> Iter 2）平均成功率由 `51.49%` 提升至 `60.33%`、`70.37%`；蒸馏后的单一 Generalist 成功率亦稳固在 `69%` 量级且无专家间干扰。

## 与知识库关联

- 实体节点：[paper-notebook-from-experts-to-a-generalist.md](../../wiki/entities/paper-notebook-from-experts-to-a-generalist.md)
- 后续工作：[cn-os-fast.md](../../wiki/entities/cn-os-fast.md)（FAST：单阶段 MoE + CoM-Aware + Parseval 残差自适应）
