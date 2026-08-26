# YIYIZH/MoeCo

> 来源归档

- **标题：** MoeCo 部分代码发布
- **类型：** repo
- **代码：** <https://github.com/YIYIZH/MoeCo>
- **论文：** [arXiv:2608.22972](https://arxiv.org/abs/2608.22972) — 归档见 [`sources/papers/moeco_arxiv_2608_22972.md`](../papers/moeco_arxiv_2608_22972.md)
- **入库日期：** 2026-08-26
- **一句话说明：** 手术三元组识别的模型、数据加载、CGL 损失与 CLIP 描述子；完整训练入口与 GMM/预提取特征声明录用后发布。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [MoeCo 实体页](../../wiki/entities/paper-moeco.md) | 方法归纳 |

## 仓库内容（README）

`dataloader.py`、`network.py`、`network_trans.py`、`loss/`、`clip/`、`run.sh` / `run_T50.sh`、`all_data*.json`。支持 CholecT45/T50 目录布局。Python 3.10 + PyTorch。

## 开源状态

**部分开源** — 网络与损失可读；README 明确「Complete runnable training/evaluation code will be released after paper acceptance」，且仍有实验机绝对路径。
