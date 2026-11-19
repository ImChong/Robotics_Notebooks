# xpeng-robotics/UniT

> 来源归档（repo）

- **标题：** UniT — Unified Physical Language for Human-to-Humanoid Transfer
- **代码：** <https://github.com/xpeng-robotics/UniT>
- **项目页：** <https://xpeng-robotics.github.io/unit/>
- **论文：** <https://arxiv.org/abs/2604.19734>
- **类型：** research-code（统一潜动作分词 + VLA / WM 下游）
- **License：** Apache-2.0
- **首次入库：** 2026-07-26

## 一句话摘要

官方实现统一潜动作分词器及其 VLA / 世界模型下游；仓库含预处理、训练/推理脚本与示例，配套 HF checkpoints。

## 可运行入口核对（2026-11-19）

| 官方入口 | 实际用途 |
|---|---|
| `examples/run_gr1_100_egodex.sh` | tokenizer 80k → EgoDex+GR1 dual-system 预训练 20k → GR1-only 微调 20k |
| `examples/run_gr1_full.sh` | tokenizer 80k → GR1 full dual-system 预训练 160k |
| `scripts/groot_tokenizer_train.py` | 共享 tokenizer 训练，输出 checkpoint |
| `scripts/dual_system_train.py` | 使用 tokenizer checkpoint 的策略预训练/微调 |
| `examples/run_eval.sh` | 启动 `scripts/inference_service_unit.py`，再使用 `scripts/simulation/simulation_service_unit.py` 在线 rollout，最后 `scripts/compute_success_rate.py` 汇总 |
| `examples/run_eval_loss.sh` | 离线动作重构 MSE 与曲线评测 |

**复现条件与边界**：README 对训练默认超参要求每卡约 120+GB 显存；仿真评测测试于 RTX 4090 24GB。公开仓库包含可运行的 VLA-UniT 路线和 HF checkpoints，未在 README 列出 WM-UniT / Cosmos 复现命令。README 参考成功率 GR1-full 66.4%、EgoDex+GR1-100 50.9%，和论文 66.7% / 50.0% 是不同报告口径。

## 仓库公开结构（入库日）

| 路径 | 作用（README / 目录名口径） |
|------|------------------------------|
| `preprocessing/` | 数据与分词相关预处理 |
| `gr00t/` | 与 GR00T 架构对接的策略侧代码树 |
| `scripts/` / `examples/` | 训练、评测与示例入口 |
| `docs/` | 文档 |

## 对 wiki 的映射

- [`wiki/entities/paper-unit-unified-physical-language.md`](../../wiki/entities/paper-unit-unified-physical-language.md)
- [`sources/papers/unit_xpeng_arxiv_2604_19734.md`](../papers/unit_xpeng_arxiv_2604_19734.md)
- [`sources/sites/xpeng-robotics-unit.md`](../sites/xpeng-robotics-unit.md)
