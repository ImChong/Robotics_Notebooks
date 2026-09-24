# FLUX 3 Action（black-forest-labs/flux-action）

> 来源归档

- **标题：** FLUX 3 Action
- **类型：** repo / world-action-model / manipulation
- **链接：** <https://github.com/black-forest-labs/flux-action>
- **HF Collection：** <https://huggingface.co/collections/black-forest-labs/flux-3-action>
- **入库日期：** 2026-09-24
- **一句话说明：** Black Forest Labs 7B 级 world action model 独立全量微调与推理包：数据准备、分布式训练、checkpoint/export、DROID/SO-101 推理；SO-101 LoRA 走 LeRobot 集成。
- **代码：** **已开源** — GitHub 仓 + 公开 HF 权重（base / DROID / SO-101；BF16 + FP8r 变体）
- **沉淀到 wiki：** [flux-3-action](../../wiki/entities/flux-3-action.md)、[lerobot](../../wiki/entities/lerobot.md)

## 核心组件（README / docs/setup.md）

| 模块 | 说明 |
|------|------|
| `transformer.py` / `transformer_inf_*` | 训练与推理分离实现 |
| HF `flux-3-action-base` | 共享冻结 video/text encoder |
| DROID policy | BF16 + FP8r；4-step guidance / GD / SD 变体 |
| SO-101 | task LoRA via [LeRobot workflow](https://github.com/black-forest-labs/flux-action/blob/main/docs/so101-lora.md) |

## 对 wiki 的映射

- [flux-3-action](../../wiki/entities/flux-3-action.md)
- 交叉 [world-action-models](../../wiki/concepts/world-action-models.md)、[lerobot](../../wiki/entities/lerobot.md)
