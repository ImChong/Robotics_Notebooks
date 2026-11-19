# weight-space-programming-autoresearch｜iWSP 官方源码与实验归档

- **仓库：** https://github.com/csinva/weight-space-programming-autoresearch
- **论文：** https://arxiv.org/abs/2611.18114 （v1，2026-11-13）
- **关联论文归档：** [iWSP](../papers/iwsp_arxiv_2611_18114.md)
- **核查日期：** 2026-11-19
- **代码公开状态：** 已公开代码、Agent 提示词、实验快照与日志，仓库标注 MIT 许可证；未发现此论文专用的预训练模型权重下载要求。
- **核对范围：** 仓库 README 的 Overview/Results/Quickstart/Reproducing/Layout，experiments/evolve/launch_claude.py、setup_run.py、interpretable_transformer.py（直接阅读入口与模型的评测分支）。
- **通读：** 已完成

## 目录与入口

- **环境：** README 要求 uv；自动 Agent 运行需 Claude Code；使用 uv sync 安装依赖。
- **Agent 启动：** uv run experiments/evolve/launch_claude.py --model opus55 --tasks gcd-three-digits --seeds 1 --workers 1
- **手工验证：** uv run experiments/evolve/setup_run.py demo，然后进入 experiments/runs/manual/demo，运行 uv run interpretable_transformer.py --task addition-five-digits。
- **实际入口调用：** launch_claude.py 的 run_one() → setup_run() 新建隔离目录/复制模板 → 生成 program.md 提示 → subprocess 调用 claude -p（restricted、限定工具） → Agent 修改 interpretable_transformer.py 中的架构/write_weights() → 允许的 Bash 命令运行该脚本 → build_model()/write_weights() → src.eval.evaluate() → 将准确率写入 results/overall_results.csv → driver 统计尝试、保存 agent_logs/metadata.json 并继续 session。
- **关键隔离：** Agent 的可调用 shell 命令限制为运行本地评测脚本，文件读写在 run 目录；代码中的最终评分属于可见集，论文 hidden set 与 audit 在实验结束后另做。源码注释明确指出只能事后审计 Agent 是否违规。
- **神经实验：** experiments/evolve-neuro/（语言 fMRI）、experiments/vision/（视觉 fMRI）、experiments/ecog/（ECoG）分别有专门 harness，不能把上述 Transformer 入口当成三项神经实验共用启动程序。
- **实验留痕：** experiments/runs/ 每个任务保存每次模型文件和分数，README 说明不使用 Git commits 作为每次试验的快照。
- **未实跑：** 只完成 README 与源码静态核对，未执行 uv sync、Agent/神经数据集运行或隐藏集评测。

## 学习与工程边界

候选特征/权重不得梯度训练，不代表神经编码的固定评测器不做岭回归拟合；论文的任务精度、真实观测数据及模型解释性须按各自实验设置对照。

- **wiki 节点：** [iWSP](../../wiki/entities/paper-interpretable-weight-space-programming.md)
- **概念对照：** [MIT Weight-Space Programming](../../wiki/entities/weight-space-programming.md)
