# Susie-Lu/driftworld

> 来源归档

- **标题：** DriftWorld（官方实现）
- **类型：** repo
- **组织 / 作者：** Susie-Lu
- **代码：** <https://github.com/Susie-Lu/driftworld>
- **权重：** <https://huggingface.co/Susie-Lu/driftworld>
- **论文：** <https://arxiv.org/abs/2607.15065>
- **项目页：** <https://susie-lu.github.io/driftworld/>
- **入库日期：** 2026-07-22
- **一句话说明：** DriftWorld **动作条件 drifting 世界模型** 官方仓：`conda env create -f driftworld/environment.yml`；当前完整覆盖 **Push-T** 训练（`main_train.py`）、可视化（`main_vis.py`）、视觉指标（`main_eval_metrics.py`）、GPC-RANK 推理时改进（`main_gpc_rank.py`）与离线策略评估（`main_policy_eval.py`）；其它数据集入口声明即将补齐。

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `driftworld/environment.yml` | conda 环境 |
| `torchrun --nproc_per_node=2 main_train.py --config-name=pushT_driftworld` | Push-T 训练（论文：2×H100） |
| `python main_vis.py` | 生成可视化视频 |
| `python main_eval_metrics.py` | MSE / SSIM / PSNR / LPIPS |
| `main_gpc_rank.py` | 推理时动作搜索（GPC-RANK） |
| `python main_policy_eval.py` | 离线策略评估 IoU 相关性 |
| `driftworld/drifting/` · `unet_multi/` | drifting loss 与动作条件 U-Net |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [DriftWorld](../../wiki/entities/paper-driftworld.md) | 实体归纳页：1-step drifting、GPC-RANK、离线评估 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 像素域 WM 谱系中的 **非扩散 / 单次前向** 分支 |
| [world-models-route-03-virtual-sandbox](../../wiki/overview/world-models-route-03-virtual-sandbox.md) | 推理时规划 + 离线策略评估沙盒 |
| [GigaWorld-1](../../wiki/entities/paper-gigaworld-1-policy-evaluation.md) | 同属「WM 作策略评估器」坐标，侧重点不同 |
| [OSCAR](../../wiki/entities/paper-oscar.md) | 同属动作条件视频 WM + 虚拟策略评估，OSCAR 偏跨具身骨架 / RoboArena |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/driftworld_arxiv_2607_15065.md`](../papers/driftworld_arxiv_2607_15065.md)
- 项目页：[`sources/sites/susie-lu-driftworld-github-io.md`](../sites/susie-lu-driftworld-github-io.md)
- 沉淀 **[`wiki/entities/paper-driftworld.md`](../../wiki/entities/paper-driftworld.md)**
