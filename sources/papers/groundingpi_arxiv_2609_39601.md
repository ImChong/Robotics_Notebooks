# GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives

> 来源归档（ingest）

- **标题：** GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives
- **类型：** paper / visual-grounding / physical-intelligence / robotics
- **论文：** <https://arxiv.org/abs/2609.39601>
- **项目页：** <https://groundingpi.github.io/>
- **代码：** <https://github.com/groundingpi/GroundingPI>
- **模型：** <https://huggingface.co/GroundingPI/GroundingPI>
- **Hugging Face 论文页：** <https://huggingface.co/papers/2609.39601>
- **作者：** Qize Yu, Lianrui Fan, Boyu Chen, Jiaqi Liang, Xini Ding, Yue Chen, Zetian Song, Yuran Wang, Yi Zou, Kaixuan Wang, Tianxing Chen, Wenxuan Song, Bohan Zhou, Mingleyang Li, Siqiao Huang, Yuqi Ye, Caigao Jiang, Wei Wei, Ruihai Wu, Hang Zhang, Yixiao Ge, Shuchang Zhou, Shilong Liu, Xianming Liu, Ping Luo, Shiyu Huang
- **提交日期：** 2026-09-30（arXiv 首次提交）
- **入库日期：** 2026-10-14
- **一句话说明：** GroundingPI 以 4B 视觉语言模型为基础，将点和框编码为共享词表中的离散坐标 token，并研究专门的视觉接地预训练如何迁移到机器人操作与自动驾驶。

## 摘要级要点

- **问题：** 通用视觉语言或视频生成骨干可能无法在拥挤场景、小目标和闭环控制需求下稳定提供精确的目标位置。
- **方法：** 训练一个 Grounding Foundation Model，让点和边界框以共享词表中的量化坐标表示；训练包含多模态与空间预训练、监督微调及基于 GRPO 的强化学习后训练，并使用公开数据及专门的数据引擎提供监督。
- **评测：** 论文报告 34 个 grounding benchmark、11 类感知能力，比较 44 个基线；平均得分为 73.68%。
- **下游迁移：** 在 RoboTwin 2.0 的四种 OOD 设定中超过论文比较的主流视觉骨干，最高相对提升 24.8%；RoboCasa-GR1 中使用 50% 示范训练的设置超过部分使用 75% 示范训练的基线；nuScenes 视觉骨干设置下报告 0.296 m 平均开环 L2 误差。
- **定位：** 该工作主张先提升面向物理任务的感知与空间 grounding 能力，再把更强的感知骨干接入动作策略；它本身不是一个直接输出机器人动作的通用策略。

## 核心摘录（面向 wiki 编译）

### 1. 共享空间 token 表示

点与边界框被表达为共享词表中的离散坐标 token。官方模型卡说明 GAM 协议使用 0–999 整数坐标网格：点为 `(x, y)`，框为 `(x1, y1, x2, y2)`。该输出便于统一处理目标定位、指代表达、OCR 与文档布局等 grounding 任务。

- **对 wiki 的映射：** [GroundingPI 实体页](../../wiki/entities/paper-groundingpi.md) — 空间表示及输入输出约定。

### 2. 分阶段训练与视觉基础模型

论文摘要将训练概括为多模态/空间预训练、监督微调和 GRPO 强化学习后训练。官方仓库进一步提供模型定义、训练配置、运行入口与评测流程；权重模型由 MoonViT-V2 视觉编码器、多模态投影器和 Qwen3-4B 语言骨干组成。

- **对 wiki 的映射：** [GroundingPI 实体页](../../wiki/entities/paper-groundingpi.md) — 方法结构、训练流程和可复现入口。

### 3. 向具身下游任务迁移

机器人操作实验包括 RoboTwin 2.0 与 RoboCasa-GR1；自动驾驶实验包括 nuScenes。论文摘要给出的指标说明 grounding 能力作为视觉骨干对下游任务有帮助，但不能据此推断模型已成为可直接部署的机器人动作策略。

- **对 wiki 的映射：** [GroundingPI 实体页](../../wiki/entities/paper-groundingpi.md) — 下游结果及动作策略边界。

## 对 wiki 的映射

- 主实体页：[wiki/entities/paper-groundingpi.md](../../wiki/entities/paper-groundingpi.md)
- 相关任务页：[wiki/tasks/manipulation.md](../../wiki/tasks/manipulation.md)
- 相关方法页：[wiki/methods/vla.md](../../wiki/methods/vla.md)

## BibTeX

```bibtex
@misc{yu2026groundingpigroundingfoundationmodel,
  title = {{GroundingPI}: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives},
  author = {Qize Yu and Lianrui Fan and Boyu Chen and Jiaqi Liang and Xini Ding and Yue Chen and Zetian Song and Yuran Wang and Yi Zou and Kaixuan Wang and Tianxing Chen and Wenxuan Song and Bohan Zhou and Mingleyang Li and Siqiao Huang and Yuqi Ye and Caigao Jiang and Wei Wei and Ruihai Wu and Hang Zhang and Yixiao Ge and Shuchang Zhou and Shilong Liu and Xianming Liu and Ping Luo and Shiyu Huang},
  year = {2026},
  eprint = {2609.39601},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2609.39601}
}
```
