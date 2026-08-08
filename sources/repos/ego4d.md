# Ego4D（facebookresearch/Ego4D）

> 来源归档（ingest）

- **标题：** Ego4D & Ego-Exo4D — dataset CLI, features, notebooks
- **类型：** repo / dataset tooling / research utilities
- **代码：** <https://github.com/facebookresearch/Ego4D>
- **项目页：** <https://ego4d-data.org/>
- **文档：** <https://ego4d-data.org/docs/>
- **论文（Ego4D）：** <https://arxiv.org/abs/2110.07058>
- **相关（同仓公告）：** Ego-Exo4D — <https://ego-exo4d-data.org/> · arXiv:2311.18259
- **许可证：** MIT（代码）
- **默认分支：** `main`
- **PyPI：** `pip install ego4d`（需 Python ≥ 3.10）
- **入库日期：** 2026-08-08
- **一句话说明：** Meta FAIR 官方仓库：Ego4D / Ego-Exo4D **下载 CLI**、视频读取抽象、特征提取（Omnivore / SlowFast 等）、notebook 与部分 research 代码（如 CLEP）；各 Ego4D benchmark 完整基线多在 [EGO4D org](https://github.com/EGO4D/) 分仓。

## 开源边界（入库日核实）

| 项 | 状态 |
|----|------|
| **代码** | **已开源（MIT）**：CLI、特征、readers、notebook、部分 research |
| **数据** | **受控开放**：需 Ego4D / Ego-Exo4D license + AWS 凭证；本仓不托管视频本体 |
| **Benchmark 训练基线** | **分散**：主仓注明完整 baseline 见 EGO4D org 与 docs；Ego-Exo4D baseline「coming soon」类说明以 README 为准 |
| **预计算特征** | 可通过 CLI / 文档获取（仓库提供提取 API） |

## 仓库结构（README 对齐）

| 路径 | 作用 |
|------|------|
| `ego4d/cli/` | Ego4D 下载 CLI（命令 `ego4d`） |
| `ego4d/egoexo/download/` | Ego-Exo4D 下载 CLI（命令 `egoexo`） |
| `ego4d/features/` | 全库特征提取 API 与 Omnivore / SlowFast 等包装 |
| `ego4d/research/` | 研究向 readers / dataloaders；含 `clep`（Contrastive Language Egocentric Pre-training）等 |
| `notebooks/` | 标注可视化与教程 notebook |
| `viz/` | narration 等可视化引擎 |
| `pyproject.toml` / `setup.py` | 可编辑安装 / PyPI 包 |

## 典型下载命令（文档对齐）

```bash
pip install ego4d --upgrade
# 签署 license 并配置 AWS 凭证后：
ego4d --output_directory="~/ego4d_data" --datasets full_scale annotations
# 或先下可视化子集：
ego4d --output_directory="~/ego4d_data" --datasets viz
```

全量 primary 视频约 **数 TB** 量级；工程上应按 benchmark / 模态筛选，勿默认拉满。

## 对 wiki 的映射

- [Ego4D 论文实体](../../wiki/entities/paper-ego4d.md) — 源码运行时序图对齐本仓库 CLI / 特征入口
- [视觉表征与策略](../../wiki/concepts/visual-representation-for-policy.md) — Ego4D 特征 / 预训练下游读法
- [Imitation Learning](../../wiki/methods/imitation-learning.md)

## 交叉链接（sources 互指）

- 项目页：[ego4d-data-org.md](../sites/ego4d-data-org.md)
- 论文：[ego4d_arxiv_2110_07058.md](../papers/ego4d_arxiv_2110_07058.md)
