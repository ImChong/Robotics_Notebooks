# Real-Time Robot Execution with Masked Action Chunking（arXiv:2601.20130）

> 来源归档（ingest）

- **标题：** Real-Time Robot Execution with Masked Action Chunking
- **短名：** REMAC
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2601.20130>
- **项目页：** <https://remac-async.github.io/>
- **代码：** <https://github.com/hatchetProject/REMAC> — [`sources/repos/remac.md`](../repos/remac.md)
- **会议：** ICLR 2026
- **机构：** 伊利诺伊大学芝加哥分校（UIC）；中佛罗里达大学（UCF）；思科研究（Cisco Research）
- **入库日期：** 2026-09-30
- **一句话说明：** 训练期 masked action chunking + prefix-preserving 采样，同时缓解异步 chunk 的 **inter-chunk 不连续** 与 **intra-chunk 感知–动作不一致**；推理无额外延迟，可与 test-time RTC 类方法叠加。

## 开源状态（步骤 2.5，2026-09-30）

- **已开源**：GitHub `hatchetProject/REMAC` 含 Kinetix 仿真两阶段管线（base flow 模仿 → LoRA REMAC 微调）；README 指向 `src_lora/train_expert.py`、`generate_data.py`、`train_flow.py` 等。

## 核心摘录（面向 wiki 编译）

- 异步 + chunking 失败不只来自边界跳变：**intra-chunk inconsistency**——执行的前缀来自旧观测 \(o_{t-h}\)，与当前 \(o_t\) 错位。
- **Prefix masking**：delay 条件 mask \(m_d=\mathbb{1}[\tau\ge d]\)，loss 只监督可执行后缀；训练随机 \(d\) 覆盖全 delay 谱。
- **Self-conditioned curriculum**：用预训练策略预测混入 flow 训练输入，\(\sigma\) 从 1 退火到 0，模拟 test-time 前缀先验。
- **Prefix-preserving sampling**：推理去噪保留已执行前缀，强化 chunk 间连续。
- 12 仿真任务 + 3 真机设定：更高成功率、更快完成、对注入 delay 更稳；可与 RTC 等 test-time 方法组合。
- **对 wiki 的映射：** [paper-remac](../../wiki/entities/paper-remac.md)
