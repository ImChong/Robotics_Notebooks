# GRAIL 项目页

> 来源归档（site / project page）

- **标题：** GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors
- **类型：** project page
- **URL：** <https://research.nvidia.com/labs/dair/grail/>
- **论文：** <https://arxiv.org/abs/2606.05160>
- **代码：** <https://github.com/NVlabs/GRAIL/>
- **数据集：** <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL>
- **机构：** NVIDIA、UCLA
- **核查日期：** 2026-07-22
- **一句话说明：** GRAIL 项目页介绍全数字人形 loco-manipulation 数据生成管线，并链接官方论文、代码和 Hugging Face 数据集。

## 核心摘录（归纳，非全文）

- 项目页称 GRAIL remains fully virtual until deployment：组合 3D assets、simulator-ready scenes、robot-proportioned characters 与 video foundation model priors。
- GRAIL 产生 **20,000+** sequences，覆盖 pick-up、whole-body manipulation、sitting、terrain traversal。
- Sim-to-real section 展示使用 GRAIL-generated data 训练 egocentric RGB policies，部署到 Unitree G1 做 object pick-up 和 stair-climbing。
- 项目页还展示 GR00T fine-tuning：95% GRAIL + 5% teleoperation mixture。
- Method section 概括三段：asset-conditioned 4D HOI generation、task-general tracking、egocentric RGB policies。

## 对 wiki 的映射

- [GRAIL 实体页](../../wiki/entities/paper-grail.md)
- [GRAIL Loco-Manipulation Dataset](../../wiki/entities/grail-locomanipulation-dataset.md)
- [Loco-Manip 接触分类 03：生成式补数](../../wiki/overview/loco-manip-contact-category-03-generative-data.md)

## 参考来源（原始）

- 项目页：<https://research.nvidia.com/labs/dair/grail/>
- GitHub：<https://github.com/NVlabs/GRAIL/>
- 数据集：<https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL>
