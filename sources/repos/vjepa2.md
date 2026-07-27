# facebookresearch/vjepa2

> 来源归档

- **标题：** V-JEPA 2 / V-JEPA 2-AC（官方实现）
- **类型：** repo
- **组织：** Meta FAIR（facebookresearch）
- **代码：** <https://github.com/facebookresearch/vjepa2>
- **License：** MIT
- **论文：** <https://arxiv.org/abs/2506.09985>
- **博客：** <https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks>
- **权重：** `https://dl.fbaipublicfiles.com/vjepa2/` + HF collection
- **入库日期：** 2026-07-27
- **一句话说明：** JEPA 视频自监督预训练与探针评测代码；含 Droid 后训练配置与 **V-JEPA 2-AC** checkpoint，支持 latent 空间动作条件预测与规划相关 notebook。

## 开源核查（2026-07-27）

| 项 | 状态 |
|----|------|
| 代码 | **已开源** · MIT |
| 预训练权重 | ViT-L/H/g 等公开 |
| V-JEPA 2-AC | `vjepa2-ac-vitg.pt` 公开 |
| Demo | `notebooks/vjepa2_demo.py` / Colab；能量景观 notebook |
| 备注 | README 亦收录 **V-JEPA 2.1**（本实体页主锚 2.0 / 2506.09985） |

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `conda create -n vjepa2-312 python=3.12` + README 依赖 | 环境 |
| `python -m notebooks.vjepa2_demo` | 加载权重跑分类样例 |
| `python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml` | 预训练入口 |
| `python -m app.main --fname configs/train/vitg16/droid-256px-8f.yaml` | Droid 上训 AC 预测器 |
| `python -m evals.main --fname configs/eval/...` | 探针训练 / 推理 |
| `notebooks/energy_landscape_example.ipynb` | AC 能量景观示例 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [V-JEPA 2](../../wiki/entities/paper-vjepa2.md) | 实体归纳：互联网预训练 + latent 规划 |
| [world-model-physics-fidelity-outputs](../../wiki/overview/world-model-physics-fidelity-outputs.md) | 相对像素视频的 latent 中间路线 |
| [DWM Separating](../../wiki/entities/paper-dwm-separating-world-effects.md) | latent WM + 规划对照 |
| [video-as-simulation](../../wiki/concepts/video-as-simulation.md) | 像素仿真动机对照 |

## 对 wiki 的映射

- 论文：[`sources/papers/vjepa2_arxiv_2506_09985.md`](../papers/vjepa2_arxiv_2506_09985.md)
- 博客：[`sources/sites/meta-vjepa2-blog.md`](../sites/meta-vjepa2-blog.md)
- 沉淀 **[`wiki/entities/paper-vjepa2.md`](../../wiki/entities/paper-vjepa2.md)**
