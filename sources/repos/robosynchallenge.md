# EDEM-AI/RoboSynChallenge

- **标题：** RoboSynChallenge 官方实现与评测框架
- **类型：** repo
- **URL：** <https://github.com/EDEM-AI/RoboSynChallenge>
- **配套论文：** [arXiv:2608.12416](https://arxiv.org/abs/2608.12416) — [`sources/papers/robosynchallenge_arxiv_2608_12416.md`](../papers/robosynchallenge_arxiv_2608_12416.md)
- **数据/权重：** <https://huggingface.co/RoboSynChallenge>
- **入库日期：** 2026-08-19
- **通读：** 已完成
- **核对范围：** 2026-11-21 核对 `EDEM-AI/RoboSynChallenge@45d9b9c`：`scripts/eval_policy.py`、`scripts/eval_openpi0_embodichain.py`、`evaluation_results/README.md`

## 一句话说明

EmbodiChain 栈上的合成数据挑战赛：sim 训练、real 评测；PI0/PI0.5/Motus + ACT/DP 基线。

## 仓库状态（2026-08-19 核查）

| 项 | 内容 |
|----|------|
| 安装 | EmbodiChain 依赖链 |
| 数据 | HF `RoboSynChallenge/datasets`（21 套） |
| 权重 | HF org `RoboSynChallenge/`（ACT/DP/SmolVLA 等） |
| 评测 | `evaluation_results/`  pinned 结果 |

## 与 wiki 的关系

- 实体页：[paper-robosynchallenge](../../wiki/entities/paper-robosynchallenge.md) — 含源码运行时序图。
- 仿真/训练底座：[EmbodiChain](../repos/embodichain.md) → [wiki/entities/embodichain.md](../../wiki/entities/embodichain.md)
