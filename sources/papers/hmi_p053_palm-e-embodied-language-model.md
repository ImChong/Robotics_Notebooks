# PaLM-E: An Embodied Multimodal Language Model（PaLM-E，HMI P053）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** PaLM-E: An Embodied Multimodal Language Model
- **短名：** PaLM-E
- **类型：** paper / hmi-papers / 世界模型、VLA与Agent
- **HMI ID：** P053
- **年份：** 2023
- **原文：** https://arxiv.org/abs/2303.03378
- **代码：** 无 / 见正文开源状态
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 把连续相机与机器人状态投影成与文本相同的嵌入序列，使视觉、状态与语言共享自回归推理上下文（输出仍主要在语言层）。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P053](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P053.md)

## 开源状态（步骤 2.5）

- **结论：** 未开源模型权重（Google 研究发布）

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

PaLM-E经常被简写成“多模态大模型做机器人”，但它真正关键的设计是输入接口：不把图像转成一句固定文字，也不为机器人单独搭一套语义网络，而是用可学习编码器把图像、3D感知和本体状态映射到LLM词嵌入空间，与文字交错排成一条“多模态句子”。

**对 wiki 的映射：** [`wiki/entities/paper-palm-e-embodied-language-model.md`](../../wiki/entities/paper-palm-e-embodied-language-model.md)

### 摘录 2

图像可由ViT等视觉编码器处理，机器人状态或3D信息也由对应编码器转成与词向量同维的嵌入。这些连续嵌入可以出现在文字指令之前、之后或中间，然后与预训练PaLM一起端到端微调。训练目标仍是预测文字token，所以跨模态迁移发生在LLM的共享表示中：视觉问答、图像描述和语言任务的数据，可以改善机器人场景的概念理解。

**对 wiki 的映射：** [`wiki/entities/paper-palm-e-embodied-language-model.md`](../../wiki/entities/paper-palm-e-embodied-language-model.md)

### 摘录 3

机器人训练样本把当前多模态观测、任务文本和期望计划/答案排成序列，统一用下一token目标优化。模型内部状态更适合表示物体关系、步骤和语义记忆，而不是精确接触动力学。它可以在新图像到来后重新生成后续文本计划，但每轮是否任务成功、物体是否抓稳以及应该调用哪个可执行技能，都要由系统提供可观察反馈。

**对 wiki 的映射：** [`wiki/entities/paper-palm-e-embodied-language-model.md`](../../wiki/entities/paper-palm-e-embodied-language-model.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-palm-e-embodied-language-model.md`](../../wiki/entities/paper-palm-e-embodied-language-model.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
