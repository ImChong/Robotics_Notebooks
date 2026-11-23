# MAPE：官方代码仓库核对

- **来源：** https://github.com/ash-ding/MAPE
- **类型：** repo / paper companion / agent memory evolution
- **核对版本：** `9d356362094b1048a05655002059a191c8ff3d22`（main，2026-11-23）
- **许可：** 项目自有代码 Apache-2.0，`baselines/*/src/` 的第三方代码另遵循自身许可（见 LICENSE/NOTICE）
- **代码：** 已公开，包括可执行搜索、评测、已进化记忆程序与消融脚本；未附运行 trace / 结果
- **论文：** https://arxiv.org/abs/2611.17068
- **对应 wiki：** [MAPE 独立论文实体](../../wiki/entities/paper-mape-agent-memory-program-evolution.md)
- **入库日期：** 2026-11-23
- **通读：** 已完成
- **核对范围：** 阅读官方 README、`programs/mape_a/README.md`、`core/memo_class.py`、`core/workflow.py`、`evolution/orchestrator.py`、`evolution/proposer.py`、`evolution/evaluator.py`、`evolution/heldout.py`、`programs/mape_a/memo.py`、`programs/mape_a/src/scaffold.py` 的接口和关键入口；核对可复现命令、数据隔离及计算预算。未执行模型服务、下载基准或逐行审计整个仓库。

## 实际可运行入口

| 入口 | 核验到的运行用途 |
|------|------------------|
| `uv run python -m evolution.orchestrator --config configs/evolution/locomo_accuracy.yaml` | 从配置启动搜索：候选提案 → sanity → 评测 → 诊断 → archive / frontier |
| `uv run python -m evolution.heldout --config configs/evaluation/mape_a_longmemeval.yaml` | 对冻结候选进行 held-out 测试，不运行 proposer |
| `uv run python scripts/run_method.py --benchmark longmemeval --n 20` | 运行已公开的 MAPE-A_LME 记忆程序小样本评测（README 命令） |
| `uv run python scripts/prepare_experiments.py` | 生成无原始轮次、无 RQS、迁移等实验配置；RQS 迁移实验另需检索 replay |
| `uv run pytest -q` | 离线测试，无模型调用 |

## 模块与真实调用关系

1. `core/memo_class.py::MemoClass` 定义三个 hook：`build_memory_from_data`、`retrieve_memory_for_query`、`compose_memory_context`（后者可选）。Answer 不在可进化程序中：由宿主评测使用固定 QA prompt/model 执行，统一 Judge 判分。
2. `core/workflow.py::BaseWorkflow` 为每个样本创建隔离的 memo 实例，驱动 Build → Retrieve → Compose → QA / Judge，并对各阶段 token 与故障录 trace。
3. `programs/mape_a/memo.py::AtomicLedgerMemo` 是论文分析的已发布 MAPE-A_LME 方案。Build 抽取原子事实笔记，保留对话轮次；Retrieve 混合 dense/BM25 与 RRF；Compose 根据意图构造上下文。
4. `programs/mape_a/src/scaffold.py` 对查询文本做规则路由、日期计算与计数/求和等确定性脚手架；`src/aggregate.py` 管汇总，`src/reason.py` 提供推理 prompt；用户端的 benchmark `question_type` 标签不会进入 memory hook，避免标签泄漏。
5. `evolution/orchestrator.py` 对 proposal 先执行 sanity check，失败时可调用 proposer 修补，再由 `evolution/evaluator.py` 在 Singularity 隔离容器评测；诊断 agent 单独读取轨迹，结果写进 archive 影响下一轮 proposal；`evolution/heldout.py` 对未参与搜索的 split 做评测。
6. 公开的是 **MAPE-A_LME 的具体实现** 与通用进化框架，不能把其他 MAPE-P / LoCoMo 最终候选的全部源码都默认为同一个 `programs/mape_a/memo.py`。

## 官方环境与限制

- Python 3.12 + uv；先 `uv sync --group dev` 与 `uv run python scripts/fetch_data.py` 下载 LoCoMo 和 LongMemEval-S。
- 评测需要 OpenAI API（默认 QA / GPT-5 mini Judge / `text-embedding-3-small`）；论文 case-study 配置使用 vLLM 提供的 Qwen3-30B-A3B-Instruct-2507，README 建议双 48GB 或单 80GB GPU；搜索 proposer 需要 Claude Code CLI / Anthropic API / Singularity（Apptainer）。
- 公开仓库无打包的结果或 trace，复现需自行准备 API 密钥、数据和依赖；本次仅完成源代码静态核对，**没有运行** `pytest`、模型评测与原论文完整复现。

## 参考资料

- [论文全文](https://arxiv.org/html/2611.17068v1)
- [代码 README](https://github.com/ash-ding/MAPE/blob/main/README.md)
- [已进化程序说明](https://github.com/ash-ding/MAPE/blob/main/programs/mape_a/README.md)
- [通用接口](https://github.com/ash-ding/MAPE/blob/main/core/memo_class.py)
- [进化入口](https://github.com/ash-ding/MAPE/blob/main/evolution/orchestrator.py)
