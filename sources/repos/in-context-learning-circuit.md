# in-context-learning-circuit 官方代码

- **仓库：** https://github.com/andrewgcodes/in-context-learning-circuit
- **论文：** https://arxiv.org/abs/2611.05344v1
- **许可：** MIT（已核对仓库 `LICENSE` 原文）
- **状态：** 已公开代码、prompt batteries、图表脚本及交互式演示；完整模型实验需另行获取模型与依赖。
- **核查日期：** 2026-11-20
- **交互讲解：** [docs/index.html](https://github.com/andrewgcodes/in-context-learning-circuit/blob/main/docs/index.html)；官方 README 说明是可离线打开的单文件 3D 分步讲解，可切换标签重复与否、查询类别、示例数量（不是在线模型推理服务）。
- **核对范围：** 官方 README、LICENSE、`docs/index.html` 的入口及结构、`icl/run.py` 的初始化/提示构造/测量与干预入口；其他模块按 README 说明整理，未做逐文件源码审计与 GPU 实验。
- **通读：** 未完成

## 官方入口

README 给出 `python icl/run.py Qwen/Qwen3-4B-Base concept12 seed=1 out=runs/example`，然后分别运行 `icl/check.py runs/example` 和 `icl/effects.py effects.tsv runs/example`。主程序载入 tokenizer 与模型、调用 `tasks.build_battery`，并使用 `hooks.Hooks` 与 `score.py` 执行测量和干预，输出 `knock.jsonl`。README 称 4B 完整 battery 在单张 H100 约一小时；默认从 `NeelNanda/pile-10k` 读取用于消融尺度的文本。

可核对的入口链：`icl/tasks.py` 构建 token 对齐的 prompt battery → `icl/run.py` 装载模型并触发前向/干预 → `icl/hooks.py` / `icl/score.py` 采样激活和指标 → `knock.jsonl` → `icl/check.py` 校验 token 对齐和不变量 → `icl/effects.py` 计算配对效应与 bootstrap 区间。更细的 `icl/allheads_law.py` 与 `icl/predict_ndemo18.py` 用于拟合读取规律及检验 18 示例外推；这些运行关系依据 README，未独立执行。

**复现边界：** README 给出的主实验需要可访问的模型权重、H100 等计算资源和依赖。`docs/index.html` 是可视化教学页面，其交互不构成对论文实验结果的独立验证。

## 对 wiki 的映射

- [上下文学习的电路](../../wiki/entities/paper-a-circuit-for-in-context-learning.md)
