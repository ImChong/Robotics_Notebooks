# InternRobotics/RoboInter

> 来源归档

- **标题：** RoboInter（中间表示操作套件官方实现）
- **类型：** repo
- **组织：** InternRobotics（上海 AI Lab 等）
- **代码：** <https://github.com/InternRobotics/RoboInter>
- **论文（1.5）：** <https://arxiv.org/abs/2607.18709>
- **论文（1.0 / ICLR 2026）：** <https://arxiv.org/abs/2602.09973>
- **项目页：** <https://lihaohn.github.io/RoboInter.github.io/>
- **数据：** <https://huggingface.co/datasets/InternRobotics/RoboInter-Data>
- **VQA：** <https://huggingface.co/datasets/InternRobotics/RoboInter-VQA>
- **权重：** <https://huggingface.co/InternRobotics/RoboInter-VLM>
- **License：** MIT
- **入库日期：** 2026-07-23
- **一句话说明：** 面向机器人操作 **中间表示** 的一体化仓：数据转换与 LeRobot dataloader、半自动标注工具、VLM 微调/评测；VLA 权重与 1.5 的 World 代码截至入库日仍待齐。

## 入口速查（对齐 README，2026-07-23）

| 路径 / 命令 | 作用 |
|-------------|------|
| `RoboInterData/` | LMDB↔LeRobot 转换、HR 视频下载、带 annotation 的 PyTorch dataloader |
| `huggingface-cli download InternRobotics/RoboInter-Data --repo-type dataset` | 拉取标注数据 |
| `RoboInterTools/` | 半自动 GUI 标注（SAM2） |
| `RoboInterData-Demo/` | Gradio 标注可视化 |
| `RoboInterVLM/.../qwen-vl-finetune/scripts/sft.sh` | Qwen2.5-VL SFT（DeepSpeed ZeRO-3） |
| `RoboInterVLM/.../infer.py` / `eval/benchmark/eval_manip/` | VLM 推理与中间表示评测 |
| `RoboInterVLA/` | **占位**：README 写代码即将发布 |
| RoboInter-World | **公开树中未见**；见 1.5 论文 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [RoboInter1.5](../../wiki/entities/paper-robointer-1-5.md) | 实体归纳：IR 数据 + VQA/VLM/VLA + World |
| [VLA](../../wiki/methods/vla.md) | plan-then-execute / F-CoT 中间表示 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | IR 控制视频条件的 RoboInter-World |
| [InternVLA-A1.5](../../wiki/entities/paper-internvla-a15-unified-vla.md) | 同组织 VLA 生态；Executor 设计参考 InternVLA-M1 |
| [LeRobot](../../wiki/entities/lerobot.md) | 数据格式目标之一 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/robointer_1_5_arxiv_2607_18709.md`](../papers/robointer_1_5_arxiv_2607_18709.md)
- 项目页：[`sources/sites/lihaohn-robointer-github-io.md`](../sites/lihaohn-robointer-github-io.md)
- 沉淀 **[`wiki/entities/paper-robointer-1-5.md`](../../wiki/entities/paper-robointer-1-5.md)**
