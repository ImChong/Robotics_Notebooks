# ali-vilab/Wan-Move

> 来源归档

- **标题：** Wan-Move（官方实现）
- **类型：** repo
- **组织 / 作者：** ali-vilab（Alibaba Tongyi Lab 等）
- **代码：** <https://github.com/ali-vilab/Wan-Move>
- **权重：** HF <https://huggingface.co/Ruihang/Wan-Move-14B-480P>；ModelScope `churuihang/Wan-Move-14B-480P`
- **基准：** HF <https://huggingface.co/datasets/Ruihang/MoveBench>
- **基座：** Wan-I2V-14B（见 [Wan2.1](https://github.com/Wan-Video/Wan2.1) / [Wan 技术报告](https://arxiv.org/abs/2503.20314)）
- **论文：** <https://arxiv.org/abs/2512.08765>
- **项目页：** <https://wan-move.github.io/>
- **许可：** Apache-2.0
- **入库日期：** 2026-07-23
- **一句话说明：** 在 Wan I2V 上实现 **latent trajectory guidance** 的运动可控视频生成：`generate.py` 跑 MoveBench / 单例轨迹，`gradio_app.py` 交互画轨迹；14B-480P 可在单卡 40GB（`--t5_cpu --offload_model True --dtype bf16`）推理。

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `huggingface-cli download Ruihang/Wan-Move-14B-480P --local-dir ./Wan-Move-14B-480P` | 拉权重 |
| `huggingface-cli download Ruihang/MoveBench --repo-type dataset --local-dir ./MoveBench` | 拉基准 |
| `python generate.py --task wan-move-i2v --size 480*832 --ckpt_dir … --mode single|multi --eval_bench` | MoveBench 评测生成 |
| `python generate.py … --image … --track … --track_visibility … --prompt …` | 单视频轨迹控制 |
| `python MoveBench/bench.py` | 汇总评测指标 |
| `python gradio_app.py --ckpt_dir … --t5_cpu --offload_model True --dtype bf16` | 本地交互 demo |
| `torchrun … --dit_fsdp --t5_fsdp` | 多卡；批量评测时设 `--ulysses_size 1` |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Wan-Move](../../wiki/entities/paper-wan-move.md) | 实体归纳：latent 轨迹引导、MoveBench |
| [Wan](../../wiki/entities/paper-wan-video.md) | 开源视频基座；Wan-Move 直接微调 I2V-14B |
| [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) | 机器人掩码 WM 对照基线之一；同属「视觉空间运动条件」 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 通用视频运动控制 → 机器人条件注入谱系 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/wan_move_arxiv_2512_08765.md`](../papers/wan_move_arxiv_2512_08765.md)
- 项目页：[`sources/sites/wan-move-github-io.md`](../sites/wan-move-github-io.md)
- 沉淀 **[`wiki/entities/paper-wan-move.md`](../../wiki/entities/paper-wan-move.md)**
