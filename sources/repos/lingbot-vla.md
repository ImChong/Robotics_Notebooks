# LingBot-VLA（1.0）

> 来源归档

- **标题：** LingBot-VLA — A Pragmatic VLA Foundation Model
- **类型：** repo
- **组织：** Robbyant（蚂蚁集团具身智能）
- **代码：** <https://github.com/robbyant/lingbot-vla>
- **项目页：** <https://technology.robbyant.com/lingbot-vla>
- **论文：** <https://arxiv.org/abs/2601.18692>（PDF：`assets/LingBot-VLA.pdf`）
- **权重：** [HF `robbyant/lingbot-vla-4b`](https://huggingface.co/robbyant/lingbot-vla-4b) / [depth 变体](https://huggingface.co/robbyant/lingbot-vla-4b-depth) / [ModelScope 集合](https://modelscope.cn/collections/Robbyant/LingBot-VLA)
- **数据集：** [GM-100 `robbyant/gm100`](https://huggingface.co/datasets/robbyant/gm100)
- **依赖骨干：** [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)、[MoGe-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal)、[LingBot-Depth](https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14)
- **入库日期：** 2026-07-17
- **一句话说明：** LingBot-VLA **1.0** 官方实现：**4B** 预训练权重（含 depth 蒸馏变体）、**2 万小时** 九类双臂真机数据、LeRobot v3.0 后训练栈与 RoboTwin/GM-100 评测脚本；后续演进见 [LingBot-VLA 2.0](lingbot-vla-v2.md)。

## 开源状态（项目页核查 2026-07-17）

- **已开源：** 训练/推理代码、**4B 预训练与 RoboTwin 后训练 checkpoint**、GM-100 数据集入口、开环/闭环评测脚本。
- **许可：** Apache-2.0。

## 对 wiki 的映射

- [LingBot-VLA](../../wiki/entities/lingbot-vla.md) — 1.0 实体页
- [LingBot-VLA 2.0](../../wiki/entities/lingbot-vla-v2.md) — 6B / 全身动作 / MoE 后继产品
- [VLA](../../wiki/methods/vla.md) — 方法总览
