# EigenDEXplore（Stanford 灵巧手结构化探索）

> 来源归档

- **标题：** hgupt3/eigendexplore
- **类型：** repo
- **来源：** Stanford（Harsh Gupta 等）
- **链接：** <https://github.com/hgupt3/eigendexplore>
- **代码：** <https://github.com/hgupt3/eigendexplore>
- **项目页：** <https://eigendexplore.github.io/>（见 [eigendexplore-project.md](../sites/eigendexplore-project.md)）
- **论文：** <https://arxiv.org/abs/2610.07681>
- **入库日期：** 2026-10-12
- **许可证：** MIT
- **一句话说明：** 论文官方代码：独立 PyTorch 分布（含 rl_games 集成）、六只手（Allegro、Inspire、Schunk、Sharpa、Wuji、XHand）的 EgoSuite 拟合基、四个基准适配器与最小示例。
- **沉淀到 wiki：** [`wiki/entities/paper-eigendexplore.md`](../../wiki/entities/paper-eigendexplore.md)

## 目录与入口（2026-10-12 浏览首页与目录，未运行）

- `packages/eigendexplore/`：独立 PyTorch 分布与 rl_games 集成
- `src/action_bench/`：基加载、手布局、动作映射、CLI、基准适配器；`assets/` 含每只手拟合的基；`presets/` 为默认基准设置
- `taskpacks/`：SPIDER 与 DeXtreme 的任务输入；`third_party/` + `scripts/setup_benchmark.py`：固定版本基准源与安装器
- `examples/ppo.py`、`examples/sampling.py`；`docs/` 含 DeXtreme、SimToolReal、SPIDER 指南

## 安装与运行

- Python 3.10+；建 venv、装 CPU torch，`pip install ./packages/eigendexplore .`；`action-bench list` 检查；`python examples/ppo.py --updates 4`、`python examples/sampling.py`。
- 各基准装在各自仿真环境，指南假设 Linux + NVIDIA GPU + Conda；录像需 `[recording]` 额外依赖与 `PyOpenGL==3.1.10`，Ubuntu 需 `libosmesa6`。
- 用法：`load_eigendexplore("allegro")` 得到噪声分布，调用 `distribution`、`sample`、`log_prob`；README 建议 IID σ 从 √0.5 起。可选 `correlation`（默认 0）不在论文中，作者称初测效果好但未验证。

## 状态

1 次提交、41 星、无 release；DexMachina 适配未见 docs 指南。
