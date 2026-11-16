# Cortex: Consolidating Context into Parametric Memory（将上下文巩固为参数记忆）

> arXiv v1 一手论文核读归档；为复核笔记，不复制论文全文。

- **论文：** https://arxiv.org/abs/2610.20399
- **PDF：** https://arxiv.org/pdf/2610.20399
- **版本与日期：** arXiv:2610.20399v1，2026-10-14（Technical report）。
- **作者：** Jintian Zhang、Feng He、Xinnian Liang、Ziwen Xu、Haoming Xu、Haojing Huang、Shuangzhi Wu、Ningyu Zhang。
- **机构：** 浙江大学（Zhejiang University）、字节跳动 Seed（ByteDance Seed）、清华大学（Tsinghua University）。
- **作者给出的代码入口：** https://github.com/zjunlp/Cortex（[源码开放核验](../repos/zjunlp-cortex.md)）。
- **入库时间：** 2026-11-16。
- **核对范围：** arXiv v1 PDF：摘要、§1–7 的定义/方法/实验/局限与图表；附录中任务设置、训练协议与持续编辑实验（含 B.1、B.3）。数值按作者原文口径摘录，未独立复现。
- **通读：** 已完成

## 研究问题与形式化

- 长上下文可使模型完成新任务，但会话上下文被移除时，这部分能力通常不能保留；作者借用 biological systems consolidation 的类比，研究把上下文指定的能力写入模型参数。
- **Memory Consolidation（MC）** 同时要求：目标任务在无上下文条件下保持增益；非目标任务能力尽可能不退化。对象是 LLM 的参数记忆，不是外部向量库检索。
- 记原模型为 M，给定上下文 C，目标查询 Q+，非目标保持集 Q−，更新后的模型 M'。评估采用：
  - Context Gain：Δctx = Perf(M,Q+,C) − Perf(M,Q+)。
  - **Consolidation Rate（CR）** = [Perf(M',Q+) − Perf(M,Q+)] / Δctx，衡量移去 C 后保住的上下文增益。
  - **Retention Rate（RR）** = Perf(M',Q−) / Perf(M,Q−)，衡量非目标能力保留比例。
- CR 是归一化增益，并非一般意义的任务准确率；极端情况下可能超过 100%。报告时需确认分母为正、非零并对齐相同任务协议。

## 三阶段方法（论文方法节）

1. **Prospective rehearsal（前瞻式排练）：** 冻结当前模型，依据上下文 C 生成后续可能被问到的查询 q；再在 C 条件下生成教师回答 y，形成蒸馏训练样本。Agent 经验任务涉及预定义的环境查询和离线成功轨迹；训练与评测查询分离，不把线上环境交互当作无限补充数据。
2. **FFN plasticity（前馈网络可塑性）：** 更新能力尽量局限于 Feed-Forward Network（FFN）。Cortex-Synaptic 对 FFN gate/up/down 使用低秩 LoRA 增量；Cortex-Structural 增加非线性通道后融合成加宽的 FFN。对于 MoE，在所有 token 可访问的共享 FFN 分支中学习新记忆，并保持路由设定不随意变动。
3. **Context-to-memory distillation（上下文至记忆蒸馏）：** 教师模型读取 C+q，学生模型只读取 q；通过 top-K 词表概率分布的前向 KL 和 FFN 后隐藏状态 MSE，使新 FFN 参数吸收教师在上下文帮助下的响应。训练混入一般保持集查询防止能力偏移。
4. **顺序更新：** 冻结先前巩固的 FFN 增量；保存旧查询、用已有模型重算教师目标，避免保存全部历史上下文及其完整回答。多轮更新仍有遗忘，不能宣传无上限知识写入。

## 测试床与适用范围

- 五类目标：ALFWorld/τ²-Bench 智能体经验、新 API 包、人格/行为引导、RWKU 知识抑制、zsRE 事实知识修改。
- 实验涵盖单次、批量、连续更新；4B–120B 的稠密或 MoE 模型，部分骨干是多模态模型，**但这组论文研究评测以文本输入为主**，未验证完整视觉—动作机器人策略。
- 多项指标同时考察 CR 与 RR 的 Pareto 取舍：记住新东西和不忘旧东西是不同目标。不同任务/骨干并非全部共享同一实验矩阵。

## 可复核数字

- **50 次顺序事实编辑、Qwen2.5-14B-Instruct：** Cortex-Structural 的最终 CR = **88.9%**、RR = **99.8%**（摘要与 §4 数值）；不能误写成任务成功率 88.9%。
- 相同设置作者报告对照 AlphaEdit 最终 CR **22.7%**、RR **96.0%**。比较应保留「同协议、同骨干、连续 50 次编辑」范围。
- 论文顺序编辑消融：未使用旧查询排练时，经过后续更新旧编辑准确度明显下降；回放旧查询可显著改善保留，但不是完整无遗忘保证。
- 作者还对比 Synaptic、Structural、注意力联合更新、MoE 共享分支、不同蒸馏损失；具体值与配置应回到 PDF 对照，不把局部消融推广成一般规律。

## 容易混淆的边界

- **与外部 Agent 记忆：** 文件/检索/向量库保留原始事实，Cortex 把上下文响应映射变成参数更新；可观测性、撤回成本和模型版本控制均不同。
- **与 ICL：** ICL 在推理时带着上下文、不更新模型参数；Cortex 学后即使拿掉 C 也希望继续回答。
- **与具身 continual learning / RSI：** 论文没有展示真实机器人动作执行、在线控制或由系统自主提议—评测—接受的递归自改进闭环。
- **源码状态：** 截至 2026-11-16，作者 GitHub 公开根目录只有 LICENSE，没有 README / train / eval 或配置；本归档没有可执行复现报告。论文中 future release 不是已经开源。

## 对知识库的映射

- [Cortex 独立论文实体](../../wiki/entities/paper-cortex-parametric-memory.md)
- [Agent 记忆与 dreaming](../../wiki/concepts/agent-memory-and-dreaming.md)：对照外部/会话记忆与权重级巩固。
- [RSI 纵深路线](../../roadmap/depth-rsi.md)：定位于「权重更新」轴的相关研究，而非已成立的 RSI。