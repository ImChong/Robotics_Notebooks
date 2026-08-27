# Visual General Intelligence: A White Paper

> 来源归档（ingest）

- **标题：** Visual General Intelligence: A White Paper
- **类型：** paper（立场白皮书 / 多视角综述）
- **venue：** arXiv preprint；源自 CVPR 2026 Workshop on Visual General Intelligence
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2608.25924>
  - PDF：<https://arxiv.org/pdf/2608.25924>
  - 工作坊：<https://cvpr2026-vgi-workshop.limitlab.xyz/>
- **机构：** 产业技术综合研究所（AIST）/ FRONTia；牛津大学 VGG；OpenAI；剑桥大学；Google DeepMind；卡内基梅隆大学；纽约大学；帝国理工学院；哈佛大学；斯坦福大学；普林斯顿大学；纽伦堡工业大学 等
- **入库日期：** 2026-08-27
- **最后更新：** 2026-08-27
- **一句话说明：** CVPR 2026 VGI 工作坊白皮书：把 **视觉通才智能（VGI）** 写成「从视觉经验涌现智能、并可能通向 AGI」的研究议程；十篇立场并不统一成一个模型，但对机器人读者把 **视频生成基座、具身闭环、Spatial AI、物理结构可编辑性** 与「VLM 挂 LLM」路径拆开。

## 开源状态（步骤 2.5，2026-08-27）

| 项 | 状态 |
|----|------|
| 工作坊页 | **已发布** — <https://cvpr2026-vgi-workshop.limitlab.xyz/>（slides / 日程 / poster） |
| 训练 / 推理代码 | **确认无代码** — 立场白皮书，项目页与 PDF 均无 GitHub / 权重 / 数据集 |
| 开源结论 | **确认未开源**（不适用：无可运行实现） |

## 核心论文摘录（MVP）

### 1) VGI 的问题陈述：视觉不是 LLM 的输入通道

- **链接：** <https://arxiv.org/abs/2608.25924> §1, §3.2, §4
- **摘录要点：**
  - **VGI** 不是更准的分类器，也不是把视觉编码器接到 LLM 上的 MLLM；它问的是：在 **耦合语言之前或与语言交互时**，视觉能否理解、预测、泛化。
  - 语言是极晚近的文化产物；视觉系统可追溯到寒武纪。白皮书用这个时间尺度论证：视觉可能是捕获世界结构的 **基础智能功能**，而不只是传感器。
  - GPT 路线证明「简单自监督目标 + 规模」可涌现能力；视觉不能直接照搬，但应追问 **何种学习策略、数据、架构、评测** 能让视觉表征迁移到未知任务与环境。
  - 当前 VLM/MLLM 把图像特征、图文对、指令微调与推理缠在一起，难以分离「来自视觉经验的能力」与「来自语言模型的能力」。白皮书故意保持 **vision-only / vision-first / language-mediated / multimodal** 多视角开放。
- **对 wiki 的映射：**
  - [paper-vgi-white-paper](../../wiki/entities/paper-vgi-white-paper.md) — 主沉淀页
  - [vlm-vln-vla-vlx-world-model-taxonomy](../../wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) — 对照「语言中介」具身栈
  - [paper-from-agi-to-asi](../../wiki/entities/paper-from-agi-to-asi.md) — 语言优先 AGI 叙事的互补坐标

### 2) 十条立场：生成、持续学习、具身、结构、Spatial AI

- **链接：** <https://arxiv.org/abs/2608.25924> §2, Table 1
- **摘录要点：**
  1. **Geirhos**：生成式视频模型即视觉基础模型；Veo 3 等可零样本做分割、关键点、超分、迷宫推理（*Video models are zero-shot learners*）。
  2. **Raghunathan**：创造性（连贯 / 结构多样 / 原创 / 有用）应成为 VGI 评测，而不只是保真度。
  3. **Asano**：VGI 应在视觉寿命中 **持续、无标签、自主** 学习，而不是预训练后冻结。
  4. **Ramanan 等**：智能将是多模态、生成、高效的；足式机器人演示仍大量是 **盲策略（本体感觉）**；触觉 / 近场视觉被低估。
  5. **Fouhey**：科学发现场景下数据不可再采、无常规 GT、仪器系统误差必须拆开。
  6. **Davison**：Spatial AI 要持久可修订的世界表征，并把图结构映射到未来细粒度并行硬件。
  7. **Du**：视觉智能即具身智能：生成世界模型作感知推断、持久场景记忆、视觉计划→低层控制、主动探索、部署后适应（SILVR / World Action Verifier）。
  8. **Wu & Wu**：看见 = 从观测反演物理世界的 **code**（实体 / 内禀 / 外禀 / 关系 / 动力学）；像素生成不等于物理理解。
  9. **Liu**：vision-native：何时看、看哪里、需要多细；视觉缩放 ≠ 语言缩放（缺符号压缩层）。
  10. **Kataoka 等**：序贯预测 + 开放生成 + 重建 三目标汇合，可能从 VFM 走到视觉智能再走到通用智能。
- **对 wiki 的映射：**
  - [paper-vgi-white-paper](../../wiki/entities/paper-vgi-white-paper.md) — 十条立场对照表
  - [generative-world-models](../../wiki/methods/generative-world-models.md)、[world-action-models](../../wiki/concepts/world-action-models.md) — 生成 / 具身世界模型落点
  - [video-as-simulation](../../wiki/concepts/video-as-simulation.md) — 「视频即仿真」与 Geirhos/Du 的交汇
  - [generative-vision-pretraining](../../wiki/concepts/generative-vision-pretraining.md) — 生成即理解的视觉基座主张

### 3) 评测议程：静态任务不够，VGI 要测可干预性

- **链接：** <https://arxiv.org/abs/2608.25924> §3.2, §4
- **摘录要点：**
  - 孤立静态任务（识别榜）不够；VGI 评测应覆盖 **迁移、持续适应、主动观测、持久世界知识、创造性、物理一致性、效率**。
  - Wu & Wu：评测必须问表征能否被 **编辑、仿真、验证、用于动作**；好看的杯子落地视频不证明质量/接触/摩擦在模型里。
  - 白皮书结论：**现在问「VGI 是否已是通向 AGI 的路」为时过早**；更近的里程碑是「智能是否已从视觉经验本身涌现」。
  - 证据应包括可迁移能力、可修订世界知识、反事实、主动寻信息、持续适应、空间/物理一致性、陌生情境下的可靠动作——而不是固定视觉任务上的更高分。
- **对 wiki 的映射：**
  - [paper-vgi-white-paper](../../wiki/entities/paper-vgi-white-paper.md) — 评测节
  - [paper-worldscore](../../wiki/entities/paper-worldscore.md) — 白皮书引用的世界生成评测（ICCV 2025）

## 对 wiki 的映射（汇总）

- [paper-vgi-white-paper.md](../../wiki/entities/paper-vgi-white-paper.md) — 主沉淀页
- [paper-from-agi-to-asi.md](../../wiki/entities/paper-from-agi-to-asi.md) — 语言优先 vs 视觉优先 AGI 路径
- [generative-world-models.md](../../wiki/methods/generative-world-models.md)
- [world-action-models.md](../../wiki/concepts/world-action-models.md)
- [video-as-simulation.md](../../wiki/concepts/video-as-simulation.md)
- [generative-vision-pretraining.md](../../wiki/concepts/generative-vision-pretraining.md)
- [vlm-vln-vla-vlx-world-model-taxonomy.md](../../wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 关联 wiki 页面的参考来源 / 关联段落已添加 ingest 链接
