# π₀.₅ 项目页：A VLA with Open-World Generalization

> 来源归档（site / 官方项目页）

- **标题：** π₀.₅: a VLA with Open-World Generalization
- **类型：** site / project-page
- **URL：** <https://www.pi.website/blog/pi05>
- **论文：** <https://arxiv.org/abs/2504.16054> · [官方 PDF](https://www.pi.website/download/pi05.pdf)
- **代码：** <https://github.com/Physical-Intelligence/openpi>
- **关联技术说明：** [Knowledge Insulation](https://www.pi.website/research/knowledge_insulation)
- **作者 / 组织：** Physical Intelligence
- **发表日期：** 2025-04-22
- **入库日期：** 2026-10-03
- **一句话说明：** 官方项目页介绍通过异构数据协同训练，提升 VLA 在训练未见家庭环境中的长时程任务泛化。

## 项目页要点

- π₀.₅ 面向在新家庭场景中执行整理、清洁等操作，重点研究开放世界环境泛化。
- 核心训练思路是把多源机器人数据、通用视觉语言资料与高层语义监督共同用于训练，使模型同时学物理技能、任务语义和高层任务结构。
- 任务跨度从物体重排到擦拭污渍等长时程行为；项目页展示的是实际家庭环境实验，作者也说明模型仍会在语义决策和动作执行中犯错。
- 论文讨论的协同训练资料可组合图像、动作、文本和多模态标注（如检测框），也包括图像描述、视觉问答、机器人示范及高层子任务监督。

## 开源核查（截至 2026-10-03）

- 项目页对应论文与官方代码仓可公开访问；[openpi README](https://github.com/Physical-Intelligence/openpi) 列出 π₀.₅ 基础权重及 LIBERO、DROID 微调 checkpoint，也给出推理与微调示例。
- **部分开源**：代码和若干 checkpoint 可获取；公开页面没有提供组成预训练数据的完整数据集。openpi README 说明 π₀.₅ 的代码路径目前支持 flow-matching head。
- Knowledge Insulation 的论文、技术说明和其在 openpi 微调实现中的边界，分别见 [项目页](https://www.pi.website/research/knowledge_insulation)、[arXiv](https://arxiv.org/abs/2505.23705) 与 [官方 issue #649](https://github.com/Physical-Intelligence/openpi/issues/649)。

## 关联归档

- 论文来源：[sources/papers/hmi_p059_pi05-open-world-vla.md](../papers/hmi_p059_pi05-open-world-vla.md)
- 官方仓库：[sources/repos/openpi.md](../repos/openpi.md)
- 论文详情：[wiki/entities/paper-pi05-open-world-vla.md](../../wiki/entities/paper-pi05-open-world-vla.md)
- Knowledge Insulation 归档：[sources/papers/knowledge_insulation_arxiv_2505_23705.md](../papers/knowledge_insulation_arxiv_2505_23705.md)
