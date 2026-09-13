# PhysBrainEvalKit

> 来源归档

- **标题：** PhysBrainEvalKit — reproducible evaluation on spatial and embodied-intelligence benchmarks
- **类型：** repo（评测工具链）
- **机构：** 机智赛博（DeepCybo）
- **链接：** <https://github.com/DeepCybo-PhysAI/PhysBrainEvalKit>
- **项目页：** <https://deepcybo-physai.github.io/PhysBrain-1.5/>
- **上游框架：** [EmbodiedEvalKit](https://github.com/pickxiguapi/EmbodiedEvalKit)
- **入库日期：** 2026-09-13
- **语言：** Python
- **许可证：** 仓内未声明 LICENSE 文件（截至核查日）
- **代码 / 开源状态：** **已开源（评测子集）** — 28 benchmark 适配器、HF 推理后端、分片 runner、点定位指标协议；**不含** 模型权重与训练代码
- **一句话说明：** PhysBrain 1.5 官方 28 项具身空间智能 benchmark 复现工具；支持 Qwen3-VL 与 PhysBrain 1.5 HF 权重批量评测。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-sa-2512-16793-physbrain-human-egocentric-data-as-a-bridge-from.md`](../../wiki/entities/paper-sa-2512-16793-physbrain-human-egocentric-data-as-a-bridge-from.md)
- **交叉归档：** [physbrain-1-5.md](./physbrain-1-5.md)、[physbrain-1-5-github-io.md](../sites/physbrain-1-5-github-io.md)

---

## 仓内结构（README 摘要）

| 路径 | 作用 |
|------|------|
| `benchmark/` | 数据集适配器与 benchmark 实现 |
| `core/` | HF 推理后端、媒体处理、共享指标 |
| `eval_*.py` | 单 benchmark CLI 入口 |
| `scripts/` | 分片 runner、`benchmark_registry.py`、分数汇总 |
| `docs/` | 点定位指标协议 |
| `AGENTS.md` | 面向 coding agent 的环境与评测 SOP |

## 最短评测路径

```bash
cd /path/to/PhysBrainEvalKit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_HOME=/data/huggingface
export HF_DATASETS_CACHE=/data/huggingface/datasets
bash scripts/eval_qwen3vl.sh \
  --model-path /path/to/PhysBrain1.5-8B \
  --model-name physbrain1.5-8b \
  --output-base /path/to/results \
  --gpus 0,1 --models-per-gpu 2 --cpu-per-worker 4 \
  --dry-run
# 去掉 --dry-run 开始评测；可加 --resume 断点续跑
```

- 默认 **greedy** 解码（`temperature=0.0`）；MMSI-Bench 例外支持采样温度。
- 模型目录须为 Hugging Face 格式（含 `config.json`、tokenizer、权重分片）。
- PhysBrain 1.5 训练使用 **FA4**；评测支持 FA2/FA4，FA2 可能有轻微波动。

---

## 对 wiki 的映射

- 实体页：[PhysBrain](../../wiki/entities/paper-sa-2512-16793-physbrain-human-egocentric-data-as-a-bridge-from.md)
- 方法交叉：[VLA](../../wiki/methods/vla.md)
