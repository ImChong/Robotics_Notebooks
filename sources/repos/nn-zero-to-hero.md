# nn-zero-to-hero（karpathy/nn-zero-to-hero）

> 来源归档

- **标题：** nn-zero-to-hero
- **类型：** repo / course-materials
- **来源：** Andrej Karpathy
- **链接：** <https://github.com/karpathy/nn-zero-to-hero>
- **配套播放列表：** [Neural Networks: Zero to Hero](../courses/karpathy_zero_to_hero_youtube.md) — <https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ>
- **入库日期：** 2026-07-12
- **一句话说明：** Zero to Hero 视频课的 **notebook 归档仓**：`lectures/` 目录按讲次保存边讲边写的 Jupyter 文件，并链到 micrograd / makemore / minbpe 等独立实现仓库。
- **为什么值得保留：** 与 [`karpathy_zero_to_hero_youtube.md`](../courses/karpathy_zero_to_hero_youtube.md) 构成 **视频 / 代码** 双入口；机器人读者跟完 #1–#7 即可衔接 [Transformer](../../wiki/concepts/transformer.md) 与 [VLA](../../wiki/methods/vla.md) 论文。
- **沉淀到 wiki：** 交叉更新 [`wiki/entities/andrej-karpathy.md`](../../wiki/entities/andrej-karpathy.md)、[`wiki/concepts/backpropagation.md`](../../wiki/concepts/backpropagation.md)

## README 要点（归纳）

- **定位：** 从基础开始的神经网络课；YouTube 上边写边训，notebook 落入 `lectures/`。
- **许可：** MIT。
- **讲次与仓库映射（官方 README）：**

| 讲次 | 主题 | Notebook 路径 | 独立仓库 |
|------|------|---------------|----------|
| 1 | micrograd / 反向传播 | `lectures/micrograd` | [micrograd](https://github.com/karpathy/micrograd) |
| 2 | makemore bigram | `lectures/makemore/makemore_part1_bigrams.ipynb` | [makemore](https://github.com/karpathy/makemore) |
| 3 | MLP | `lectures/makemore/makemore_part2_mlp.ipynb` | makemore |
| 4 | Activations / BatchNorm | `lectures/makemore/makemore_part3_bn.ipynb` | makemore |
| 5 | 手推 Backprop | `lectures/makemore/makemore_part4_backprop.ipynb` | makemore |
| 6 | WaveNet 式 CNN | `lectures/makemore/makemore_part5_cnn1.ipynb` | makemore |
| 7 | 从零写 GPT | 见视频描述 | [nanoGPT](https://github.com/karpathy/nanoGPT) |
| 8 | GPT Tokenizer / BPE | 见视频 + Colab | [minBPE](https://github.com/karpathy/minbpe) |

- **播放列表第 8 讲**（State of GPT 演讲）与 **第 10 讲**（GPT-2 124M 复现）在 README 标注为 ongoing / 见视频描述；完整 10 集目录见 [`karpathy_zero_to_hero_youtube.md`](../courses/karpathy_zero_to_hero_youtube.md)。

## 对 wiki 的映射

- [`wiki/entities/andrej-karpathy.md`](../../wiki/entities/andrej-karpathy.md)
- [`wiki/concepts/backpropagation.md`](../../wiki/concepts/backpropagation.md)
- [`wiki/concepts/transformer.md`](../../wiki/concepts/transformer.md)
- [`wiki/entities/llms-from-scratch-raschka.md`](../../wiki/entities/llms-from-scratch-raschka.md)

## 参考来源（原始）

- GitHub README：<https://github.com/karpathy/nn-zero-to-hero>（2026-07-12 抓取要点）
- 课程页：<https://karpathy.ai/zero-to-hero.html>
