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

- 论文：[`sources/papers/vjepa2_arxiv_2506_09985.md`](../papers/vjepa2_arxiv_2506_09985_full_reading_2026_11_15.md)
- 博客：[`sources/sites/meta-vjepa2-blog.md`](../sites/meta-vjepa2-blog.md)
- 沉淀 **[`wiki/entities/paper-vjepa2.md`](../../wiki/entities/paper-vjepa2.md)**

## 2026-11-15 原版与当前入口复核

通读当前 README、`app/main.py` / `scaffold.py`、Droid YAML、能量景观 notebook 与 `notebooks/utils/world_model_wrapper.py` / `mpc_utils.py`，核对 `app/vjepa_droid/train.py` 的模型构建和 `train_step`。当前树 SHA 为 `204698b45b3712590f06245fbfba32d3be539812`；源码阅读未运行 GPU 或真机。

- YAML 的 `app: vjepa_droid` 经 scaffold 动态分派到训练模块，权重和 Droid CSV 需替换本地路径。逐帧目标编码在 `no_grad` 中执行，预测器优化 teacher-forcing 与两步自回归 L1。
- 公开配置 `dataset_fpcs: [8]` 与论文叙述的 16 帧输入分别记录。notebook 的 CPU 搜索用 25 samples × 2 轮、H=2，论文真机用 800×10、H=1，不能等同计时或质量。
- `WorldModel.infer_next_action` 调 `cem` 并通过 predictor 闭包滚出 latent；输出动作不代表已实现真机驱动。当前 xyz 是按轴裁剪、旋转维度固定为零，不是严格 L1-ball 投影。
- 2.1 配方与原版是不同论文和权重系列。本节点只核对原版 arXiv:2506.09985；主体 MIT，README 指定的少量增强/worker 文件为 Apache 2.0。

全部论文结果、子目标条件、相机与时延边界见[主论文通读记录](../papers/vjepa2_arxiv_2506_09985_full_reading_2026_11_15.md)。
