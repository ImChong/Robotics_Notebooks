# in-context-learning-circuit 官方代码

- **仓库：** https://github.com/andrewgcodes/in-context-learning-circuit
- **论文：** https://arxiv.org/abs/2611.05344v1
- **许可：** MIT（仓库 README）
- **状态：** 已公开代码、prompt batteries、图表脚本及交互式演示；完整模型实验需另行获取模型与依赖。
- **核查日期：** 2026-11-06

## 官方入口

README 给出 `python icl/run.py Qwen/Qwen3-4B-Base concept12 seed=1 out=runs/example`，然后分别运行 `icl/check.py runs/example` 和 `icl/effects.py effects.tsv runs/example`。主程序载入 tokenizer 与模型、调用 `tasks.build_battery`，并使用 `hooks.Hooks` 与 `score.py` 执行测量和干预，输出 `knock.jsonl`。README 称 4B 完整 battery 在单张 H100 约一小时；默认从 `NeelNanda/pile-10k` 读取用于消融尺度的文本。

已核读 README 和 `icl/run.py` 的初始化、battery、打分和干预入口；其他相关模块尚需逐项核读，图中后处理关系据 README，未运行复现。

## 对 wiki 的映射

- [上下文学习的电路](../../wiki/entities/paper-a-circuit-for-in-context-learning.md)
