# Effective-Degree（xinzaixinzai/Effective-Degree）

> 来源归档（repo）

- **标题：** Effective-Degree — Quantifying and Optimizing Simplicity via Polynomial Representations
- **类型：** repo / ML / generalization / regularization / ICML-2026
- **来源：** xinzaixinzai（GitHub）
- **链接：** <https://github.com/xinzaixinzai/Effective-Degree>
- **论文：** [arXiv:2605.29823](https://arxiv.org/abs/2605.29823) · ICML 2026 — 归档见 [`sources/papers/effective_degree_arxiv_2605_29823.md`](../papers/effective_degree_arxiv_2605_29823.md)
- **Stars：** ~12（2026-08-12）
- **入库日期：** 2026-08-06；**复核：** 2026-08-12
- **一句话说明：** 官方实现 Effective Degree（ED）度量与可微正则：CIFAR/ImageNet 相关实验、grokking、CLIP wise-ft、BERT GLUE、Procgen PPO。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-effective-degree.md`](../../wiki/entities/paper-effective-degree.md)

---

## 开源状态（步骤 2.5）

| 项 | 状态（2026-08-12 复核） |
|----|-------------------|
| 训练 / 评测代码 | **已开源**（根目录与 `poly/`、`rl/`、`bert/`、`wise-ft/`、`grokking/`） |
| 环境文件 | `environment.yaml`（主实验）、`environment_rl.yaml`（RL） |
| ImageNet Model Soups 权重 | 需另从 [mlfoundations/model-soups](https://github.com/mlfoundations/model-soups) 下载 |
| 许可证 | 仓库未声明 SPDX license（截至 2026-08-12） |
| 独立项目页 | 无；入口为 GitHub README |

**结论：** **已开源可运行实现**；大规模 ImageNet 相关实验依赖外部权重/数据准备。

---

## README 宣称的技术栈 / 入口

| 组件 | 路径 / 命令 |
|------|-------------|
| ED 正则训练（CIFAR ViT） | `train_wd_regular_torch.py` + `run.sh` |
| 相关实验（ResNet / ViT-Tiny） | `corr_resnet.sh` / `corr_vit_tiny.sh` → `poly/eval_abd.sh` |
| Grokking | `grokking/scripts/train_grokk.py` |
| CLIP FT | `wise-ft/run.sh` |
| GLUE | `bert/run_glue_reg.sh` |
| RL | `rl/ppo_procgen.sh` / `rl/ppo_procgen.py` |
| 核心度量实现 | `poly/weighted_degree.py`、`poly/wd_regularization_torch.py` |

## 关联资料

- 论文归档：[`sources/papers/effective_degree_arxiv_2605_29823.md`](../papers/effective_degree_arxiv_2605_29823.md)
- Wiki 实体：[wiki/entities/paper-effective-degree.md](../../wiki/entities/paper-effective-degree.md)
