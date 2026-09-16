# smolvla-libero-onnx

> 来源归档（ingest）

- **仓库：** <https://github.com/rafiqul713/smolvla-libero-onnx>
- **简称：** smolvla-libero-onnx
- **类型：** repo / vla / deployment / libero / onnx
- **论文：** [arXiv:2609.14146](https://arxiv.org/abs/2609.14146) — [`sources/papers/smolvla_onnx_libero_arxiv_2609_14146.md`](../papers/smolvla_onnx_libero_arxiv_2609_14146.md)
- **许可证：** MIT
- **入库日期：** 2026-09-16
- **一句话说明：** SmolVLA×LIBERO 的 PyTorch/ONNX 导出、延迟基准、闭环评测与语言宽度消融脚本；结果 JSON 入库。

## 开源状态（步骤 2.5，2026-09-16）

**已开源** — `scripts/`（export、latency、LIBERO eval、parity、lang-width ablation）、`results/baseline_summary.json`、`docs/REPRODUCE.md`。`exports/` ONNX（~2+ GB）**不包含**，需 tether 本地重导出。

## 目录要点（README）

| 路径 | 内容 |
|------|------|
| `scripts/run_parity_smoke.sh` | 快速 parity 烟测 |
| `scripts/run_lang_width_ablation.sh` | 语言宽度长消融 |
| `results/baseline_summary.json` | 主表数字 |
| `results/lang_ablation/summary.json` | 宽度消融汇总 |

## 交叉链接

- Wiki：[paper-smolvla-onnx-libero](../../wiki/entities/paper-smolvla-onnx-libero.md)
- 论文摘录：[smolvla_onnx_libero_arxiv_2609_14146.md](../papers/smolvla_onnx_libero_arxiv_2609_14146.md)
