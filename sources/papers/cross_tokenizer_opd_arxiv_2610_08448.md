# Rethinking Cross-Tokenizer On-Policy Distillation: From Alignment Coverage to Supervision Reliability

> 来源归档（paper）

- 类型：paper / on-policy distillation / tokenizer alignment
- arXiv：https://arxiv.org/abs/2610.08448v1
- 作者：Bingxi Hou、Guochao Jiang、Guofeng Quan、Weiqing Li、Wenfeng Feng、Guohua Liu、Yuewei Zhang
- 入库日期：2026-10-11
- 说明：比较跨 tokenizer 的严格位置对齐与错配 span 监督；arXiv 页面未列官方代码仓库。

## 摘录与 wiki 映射

- 三组异构 teacher–student 配对，任务覆盖数学推理和代码生成。
- 严格 1:1 token 组覆盖多数 student tokens，严格位置的共享词表保留几乎全部 teacher/student 概率质量。
- student 选择共享词表 top-16 做 reverse KL，准确率接近全共享词表 OPD，优于文中跨 tokenizer 基线。
- 在错配 span 加 log-probability MSE 虽增加监督覆盖，却降低准确率；其梯度与严格损失弱一致或负一致。
- 对 wiki 的映射：[Cross-Tokenizer OPD 实体页](../../wiki/entities/paper-cross-tokenizer-opd.md)，与 [SRD](../../wiki/entities/paper-self-retrospection-distillation.md) 和 [Shallow-π](../../wiki/entities/paper-shallow-pi.md) 对照。
