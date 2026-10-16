# GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed

> 来源归档（ingest）

- **标题：** GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed
- **类型：** paper / visual-grounding / diffusion-language-model / computer-vision
- **论文：** <https://arxiv.org/abs/2609.39600>
- **项目页：** <https://groundingpi.github.io/groundanything/>
- **代码：** <https://github.com/groundingpi/GroundAnything>
- **模型权重（DLM）：** <https://huggingface.co/GroundingPI/GroundAnything>
- **模型权重（自回归 VLM）：** <https://huggingface.co/GroundingPI/GroundAnything-VLM>
- **Hugging Face 论文页：** <https://huggingface.co/papers/2609.39600>
- **作者：** Qize Yu, Lianrui Fan, Bowen Ping, Xini Ding, Zetian Song, Junbo Niu, Kaixuan Wang, Tianxing Chen, Yue Chen, Minghua He, Yuran Wang, Jie Huang, Haojun Zhang, Min Chen, Hao Li, Wenxuan Song, Ruihai Wu, Xianming Liu, Shilong Liu, Shuchang Zhou, Ping Luo, Shiyu Huang
- **机构：** XPeng Inc.; Peking University; The University of Hong Kong; University of California, Berkeley; Princeton University; National University of Singapore; HKUST (GZ)
- **提交日期：** 2026-09-30（arXiv 首次提交）
- **入库日期：** 2026-10-16
- **一句话说明：** GroundAnything 将视觉 grounding 视作由图像与查询共同约束的空间证据提取任务，以双向扩散的分块去噪并行生成空间预测，同时保留自回归版本。

## 摘要级要点

- **问题：** 自回归 grounding 将空间预测按 token 顺序串行生成，引入延迟并强加并非任务本身所需的因果顺序。
- **方法：** 4B 参数模型先进行公开数据与专用数据引擎上的 grounding 预训练，再执行自回归到扩散模型转换、AR 与扩散联合目标训练、监督微调和 GRPO 强化学习后训练；扩散模型通过分块去噪并行提出并迭代精炼空间假设。
- **评测：** 论文覆盖 30 个 grounding benchmark。摘要报告自回归 GroundAnything-VLM 总体得分为 72.42%，并称其在同规模模型中达到新的总体最佳；GroundAnything 的熵引导解码平均为 61.75%，LocateAnything 的 MTP 快速基线为 53.32%。
- **推理速度：** 可选自推测解码相较自回归对应版本实现 4.51× 加速，COCO F1mIoU 下降 0.74 个百分点；该数字对应论文所述评测设置，不应视为任意硬件或服务负载下的保证。
- **定位：** GroundAnything 输出视觉定位结果，适合作为感知组件；论文没有把它定义为直接输出机器人动作的 VLA 策略。

## 核心摘录（面向 wiki 编译）

### 1. Grounding 可用并行空间预测表达

论文将对象、位置和空间关系视作被图像与查询共同约束的视觉证据。它们彼此相关，但并不自然要求按左到右顺序逐 token 产生，因此双向扩散可以并行提出空间假设，再迭代细化。

- **对 wiki 的映射：** [GroundAnything 实体页](../../wiki/entities/paper-groundanything.md) — 任务形式化与分块扩散解码。

### 2. 扩散与自回归模型提供不同推理路径

GroundAnything 的主模型使用熵引导的扩散解码，并支持自推测解码；GroundAnything-VLM 是独立的自回归 checkpoint。对速度、精度或运行时的比较应注明具体 checkpoint 与解码策略。

- **对 wiki 的映射：** [GroundAnything 实体页](../../wiki/entities/paper-groundanything.md) — DLM / VLM 版本与速度—精度权衡。

### 3. 速度结果依赖解码和基础设施

论文既分析熵阈值、候选块长等解码参数，也研究渐进式推理基础设施优化。自推测加速与 COCO F1mIoU 变化来自特定实验设置；部署评估需同时记录硬件、并发、服务配置和评测协议。

- **对 wiki 的映射：** [GroundAnything 实体页](../../wiki/entities/paper-groundanything.md) — 解码指标、工程边界及复现路径。

## 对 wiki 的映射

- 主实体页：[wiki/entities/paper-groundanything.md](../../wiki/entities/paper-groundanything.md)
- 相关实体页：[wiki/entities/paper-groundingpi.md](../../wiki/entities/paper-groundingpi.md)
- 相关概念页：[wiki/concepts/visual-foundation-model-trends.md](../../wiki/concepts/visual-foundation-model-trends.md)

## BibTeX

```bibtex
@misc{yu2026groundanythingreconcilingparalleldecoding,
  title = {GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed},
  author = {Qize Yu and Lianrui Fan and Bowen Ping and Xini Ding and Zetian Song and Junbo Niu and Kaixuan Wang and Tianxing Chen and Yue Chen and Minghua He and Yuran Wang and Jie Huang and Haojun Zhang and Min Chen and Hao Li and Wenxuan Song and Ruihai Wu and Xianming Liu and Shilong Liu and Shuchang Zhou and Ping Luo and Shiyu Huang},
  year = {2026},
  eprint = {2609.39600},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2609.39600}
}
```
