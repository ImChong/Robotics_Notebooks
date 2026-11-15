# OptiMer / nict-astrec-att/optimer 官方代码归档

- **类型：** repo / model merging / Optuna / CPT
- **仓库：** https://github.com/nict-astrec-att/optimer
- **入口：** https://github.com/nict-astrec-att/optimer/blob/main/optuna_merge_search.py
- **README：** https://github.com/nict-astrec-att/optimer/blob/main/README.md
- **论文：** https://aclanthology.org/2026.emnlp-main.1090/
- **项目页：** https://shyyhs.github.io/optimer/
- **授权：** MIT（官方 LICENSE）
- **核对日期：** 2026-11-15
- **核对范围：** 通读官方 README 和仓库唯一主脚本 optuna_merge_search.py 的参数解析、merge YAML、mergekit 子进程、lm-eval 子进程、Optuna objective、GPU 池、top-K 重评估及输出保存。
- **开放状态：** **部分开源**；公开的是给定 base/IT/CPT checkpoint 后的自动合并搜索流程，**未提供完整 CPT 训练入口和论文所有所需模型资产**。本次只进行源码静态阅读，未运行 GPU 任务。

## 源码实际运行链

| 函数或命令 | 功能与核实要点 |
|---|---|
| `main()` / `parse_args()` | 解析 `--cpt-models`、`--base-model`、`--it-model`、`--work-dir`、`--sampler` 等；构建持久化 SQLite Optuna study |
| `optuna.samplers.TPESampler` | 默认采样器；还支持 `CmaEsSampler` 和 `RandomSampler` |
| `MergeObjective.__call__` | 对 IT 与每份 CPT checkpoint 分别采样合并权重；从 GPU queue 独占领用 GPU |
| `make_merge_yaml()` | 建立 mergekit 配置：以 base 模型为基，IT/CPT 向量按各自权重合并，过滤 embedding、head、rotary 等相关参数 |
| `run_merge()` | 通过 `subprocess.run` 调用 `mergekit-yaml <yaml> <output> --cuda --lazy-unpickle --low-cpu-memory` |
| `run_eval()` / `compute_score()` | 调用 `lm_eval`，按任务优先级抽取指标并缩放到 0–100，取 proxy task 均分 |
| `study.optimize()` | 按 GPU 数并发试验；结果持久化到 `work-dir/optuna.db`；失败可恢复但中间权重文件可清理 |
| `run_top_k_full_eval()` | 对最高分 top-K trials 重新合并、评测；输出 `top_k_summary.json` |
| `main()` 输出 | `search_summary.json`、各 trial 日志与评测目录 |

## README 示例（需自行准备模型）

```bash
python optuna_merge_search.py \
  --cpt-models /path/to/japanese_ckpt /path/to/math_ckpt \
  --base-model google/gemma-3-27b-pt \
  --it-model google/gemma-3-27b-it \
  --work-dir /path/to/optuna_work \
  --proxy-tasks ja_leaderboard_mgsm,gsm8k \
  --n-trials 50 \
  --gpus 0,1,2,3
```

- 官方 README 指出需要 Python 3.11、Optuna 4.7.0、mergekit 固定提交、lm-eval 0.4.9.1、torch 2.6.0（CUDA 12.4）等。
- **论文与脚本默认值区别：** 论文主实验搜索 100 trials，proxy task 每项先看 100 题、top-K 再看 300 题；仓库 CLI 默认 `--n-trials 50`、`--proxy-limit 300`、`--top-k-limit 100`，直接默认运行不能认为复现论文表格。
- 脚本 `run_eval` 设置 `HF_ALLOW_CODE_EVAL=1` 和 `--confirm_run_unsafe_code` 以运行代码基准；复现时应在隔离环境执行，确认所选基准和依赖可信。
- 论文的 15–35× 搜索开销下降**不包括**先行 CPT；工程落地应同时计入单域 checkpoint 训练、存储、每次合并及评测的成本。

## 对 wiki 的映射

- [OptiMer 论文实体](../../wiki/entities/paper-optimer-optimal-distribution-vector-merging.md)
- [论文原文归档](../papers/optimer_arxiv_2603_28858.md)
- [官网核查](../sites/optimer_shyyhs_github_io.md)
