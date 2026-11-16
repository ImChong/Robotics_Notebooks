# CIALLO: Learning Shared Intent Representations for Cross-Embodiment Manipulation / 学习跨本体操作的共享意图表示

> 原始论文来源归档；这里只归纳论文的公开方法、数据与评测边界，不替代全文。

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.13732
- **全文：** https://arxiv.org/html/2611.13732v1
- **版本：** arXiv:2611.13732v1（2026-11）
- **核对范围：** HTML 全文：Introduction、Related Work、Method 3.1–3.6、Experiments 4.1–4.7、Conclusion 和 LLM Usage Statement；Appendix A–E（归一化、codec、预训练、校准、动作头延迟及真机任务）。
- **通读：** 已完成
- **收录日期：** 2026-11-12

## 论文概要

CIALLO 让不同机器人的动作先映射到共享短程意图空间，再利用本体条件动作解码器控制新机器人。

## 机制要点

- 用动作、视觉变化和语言共训一个共享短程意图表示，并由本体特定解码器恢复动作。
- 论文区分新机器人短校准与下游少量任务示范；报告每任务 20 条示范的真机平均成功率 76.7%。
- 公开完整训练/推理代码入口尚未核实，实验需进一步复现。

## 评测边界

论文报告新本体至多 10 秒记录校准，真实 SO-101/Piper 测试每任务 20 条示范时平均成功率 76.7%（对比指定的 π0 微调基线 70%）；LIBERO 97.3%、LIBERO-Plus 80.8%，RoboTwin-2.0 52.7%。不同基准的训练数据与评测设置不相同，不应混作统一跨本体泛化率。

## 通读原文：方法、实验与附录复核

### 共享意图与损失

8 类本体的 action chunk 经过时间戳与 axis/arm embedding 编码为 **K=2×D=512** 的 latent intent；本体 essential vector **128 维**。视觉支路用冻结 DINOv2 比较当前/未来帧，语言支路用冻结 BERT 再由当前视觉接地。三支路共享动作解码器，以 action reconstruction + 语句遮挡重建 + 三组 InfoNCE 对比损失学习。decoder 为 TransConv+256 ordered bins 的 softmax expectation readout。

### 预训练、适配与真机协议

附录 B：**125,434** 条 episode、8 本体、16,000 steps、8×H20 约 15.7h；新本体用单条 **≤10s** 记录只更新 essential / axis / arm embeddings（约 174k 参数，1×H20 **约 30 分钟**）。随后**冻结 codec**，仅在目标任务 20 demos/task 上训练 Qwen3.5-4B 与 intent projector；真实 SO-101 与单臂 Piper 三任务、每任务 20 次评测。表 V：**76.7%** vs π0 **70.0%**，优势主要来自 wooden-fish，水瓶任务 π0 反而更强。

### 仿真与限制

LIBERO **97.3%**（论文提示部分 baseline 的统计协议不同），LIBERO-Plus **80.8%**，RoboTwin2.0 **52.7%**（50 tasks 单 checkpoint）。动作头表 XI：H20 batch1 下 CIALLO PyTorch **4.89ms**，JAX **1.46ms**；OFT **0.37ms** 更快，故不能宣传为所有头最快。作者明确未预训练专门预测 CIALLO intent 的 generalist policy；真机两个平台、三个任务，样本覆盖有限。

## 仍待独立验证

- 论文成功率与延迟属于作者报告结果，尚未在本地复现实验；需检查代码和数据版本是否可获取。

## 对 wiki 的映射

- [CIALLO](../../wiki/entities/paper-ciallo.md)
