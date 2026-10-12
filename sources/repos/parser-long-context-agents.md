# PARSER（cuhk-parser/PARSER）

> 来源归档

- **标题：** PARSER
- **类型：** repo
- **来源：** 香港中文大学（CUHK）
- **代码：** <https://github.com/cuhk-parser/PARSER>
- **项目页：** <https://cuhk-parser.github.io/>（见 [sites 归档](../sites/parser-long-context-agents.md)）
- **论文：** <https://arxiv.org/abs/2609.06702>（见 [papers 归档](../papers/parser_arxiv_2609_06702.md)）
- **许可：** MIT
- **入库日期：** 2026-10-12
- **一句话说明：** PARSER 官方实现：全异步 RL 训练主导智能体（冻结子智能体经 SGLang Router 服务）及 HotpotQA / 2Wiki 评测脚本。
- **沉淀到 wiki：** [`wiki/entities/paper-parser-parallel-read-deep-reason.md`](../../wiki/entities/paper-parser-parallel-read-deep-reason.md)

---

## 核查结论（2026-10-12 浅克隆浏览）

- **已开源：** 含 `README.md`、`LICENSE`、`chat_template.jinja`、`eval/`、`train/`、`workflow.png`。
- **目录：** `train/`（`train_4b.sh`、`run_fully_async_parser_4b.sh`、`env_setup.sh`、`set_sglang_router.sh`、`scripts/parser_agent_loop.py` / `subagent.py` / `reward.py` / `prompt.py`，内置 `verl/` 与 `Megatron-LM/`）；`eval/`（`eval_hqa.sh`、`eval_2wiki.sh`、`eval.py`、`subagent_sp.py`、`sglang.sh`、`preprocess_eval_data.py`）。
- **发布物：** HF 4B 检查点 `inNexus/parser-qwen35-4b`、数据集 `inNexus/parser_dataset`（均核实可访问）；9B 检查点未见。
- **环境：** Python 3.12、CUDA 12.8/12.9、conda 环境 `mcore`；需给 `sglang_router` 打一行导入补丁（`sglang==0.5.10.post1`）、安装 NGINX；把仓库的 chat template 拷到主导智能体检查点。
- **评测入口默认：** 子智能体 Qwen3.5-4B、`CHUNK_MAX_LENGTH=4096`、1024 条样本。
