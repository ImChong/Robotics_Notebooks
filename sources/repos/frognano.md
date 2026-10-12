# microsoft/FrogNano 仓库归档

- **类型：** GitHub 仓库
- **链接：** <https://github.com/microsoft/FrogNano>
- **代码：** <https://github.com/microsoft/FrogNano>
- **模型权重：** <https://huggingface.co/microsoft/FrogNano-4B-2609>（MIT，非 gated，2026-10-02 更新，核查时下载 1348）
- **许可：** MIT（Microsoft）
- **核查日期：** 2026-10-12（浅克隆 HEAD）
- **项目页：** [frognano 站点归档](../sites/frognano.md)

## 开放范围

- **已发布：** Leaf 框架（`frognano/harness/leaf/` 下 agent.py、environment.py、tool_runner.py、tools.py）、评测 CLI `frognano-eval`（`frognano/cli.py`，runner.py）、Kubernetes / Python 运行时（`frognano/runtimes/`）、四个基准预设（`frognano/configs/eval/` 下 swebench-verified、swebench-pro、terminal-bench-2-verified、patch-eval-verified）、测试。
- **未发现：** TaskPilot 任务合成流水线、RL（DPPO）训练代码、合成任务数据集；仓库内除 README 外无 TaskPilot 相关文件。

## 运行要点（取自 README）

- Python ≥3.12；需要 Kubernetes 集群与命名空间权限；需要 OpenAI 兼容端点（Qwen3.5 + SGLang 用 `--reasoning-parser qwen3`、`--tool-call-parser qwen3_coder`）。
- 评测预设：SWE-bench Verified 500 任务、SWE-bench Pro 731、Terminal-Bench 2.0 Verified 89、PatchEval Verified 230；三个种子、150 个共享 worker、150 步、131,072 token 上下文、temperature 0.6、top_p 0.95、top_k 20。
- 与论文分数对齐需要相同检查点、分词器、服务配置、任务镜像与评测协议。

## 对 wiki 的映射

- [FrogNano 论文详情](../../wiki/entities/paper-frognano-4b-coding-agent.md)
- [论文归档](../papers/frognano_arxiv_2609_07925.md)
