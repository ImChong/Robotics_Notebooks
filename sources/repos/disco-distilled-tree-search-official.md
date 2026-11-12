# DISCO 官方代码：Distilled Tree Search over Continuous Actions

- **官方仓库：** <https://github.com/kallol-saha/DISCO>
- **论文：** <https://arxiv.org/abs/2610.25785>
- **项目：** <https://distilled-search.github.io/>
- **模型：** <https://huggingface.co/kallol-saha/disco>
- **数据：** <https://huggingface.co/datasets/kallol-saha/disco-data>
- **核对日期：** 2026-11-12
- **核对范围：** 官方 README、pretrain.py、run_disco.py 和列示的 disco/search、disco/data.py、flowmatch 入口；不声称逐行审阅全部第三方依赖及子模块，不声称已运行所有仿真。
- **通读：** 已完成

## 源码运行时序依据

1. `pretrain.py` 处理单物体抓放数据，调用 `run_disco.train` 分别训练 grasp/place policy；单步 BC 是初始网络。
2. `run_disco.py` 按 `config/run_disco.yaml` 合并 CLI 参数并调度 `disco/search/run.py` 子进程；搜索实现按 `disco/search/mcts.py`、`expansion.py` 与 `candidates.py` 在 ManiSkill + cuRobo 上扩展和回传树信息。
3. `disco/data.py` 对搜索树产出 policy/value 监督；`run_disco.train` 调 `disco.flowmatch.train_policy`、`train_value` 训练。
4. `demo.py --task shelf --split test` 是部署形式的无搜索评测，使用发布的 policy-value checkpoint。
5. README 环境：Linux / Python 3.10 / CUDA 12.6 / 2× RTX 4090，运行需要外部依赖及 HF 模型与数据。这一复现机器配置不是论文中训练时间比较的统一硬件。

## 与 Wiki 对接

[DISCO 实体页](../../wiki/entities/paper-distilled-tree-search-multi-object-rearrangement.md)
