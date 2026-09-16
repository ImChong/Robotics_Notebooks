# SmolVLA ONNX LIBERO 部署审计（arXiv:2609.14146）

> 来源归档（ingest）

- **标题：** When Faster VLA Deployment Changes Closed-Loop Behavior: Task Success-Latency Analysis of SmolVLA Across PyTorch and ONNX Variants
- **简称：** SmolVLA-ONNX-LIBERO
- **类型：** paper / vla / deployment / libero / onnx
- **arXiv：** <https://arxiv.org/abs/2609.14146>
- **PDF：** <https://arxiv.org/pdf/2609.14146>
- **代码：** <https://github.com/rafiqul713/smolvla-libero-onnx> — 归档见 [`sources/repos/smolvla-libero-onnx.md`](../repos/smolvla-libero-onnx.md)
- **入库日期：** 2026-09-16
- **一句话说明：** RTX 2060 上审计 SmolVLA×LIBERO 的 PyTorch+AMP vs ONNX 闭环：更快不一定更高成功率；语言 token 宽度与图审计是关键。

## 开源状态（步骤 2.5，2026-09-16）

| 组件 | 状态 |
|------|------|
| GitHub `rafiqul713/smolvla-libero-onnx` | **已开源** MIT；脚本 + `results/` 实测 JSON |
| ONNX 权重（~2+ GB） | **未随仓分发** — README 要求 tether 重导出 |

**结论：已开源（评测与复现脚本）** — 大体积 ONNX 需本地导出。

## 核心摘录

### 摘录 1：主评测（100 episode/suite，seed 42）

- 底座：`HuggingFaceVLA/smolvla_libero`；**MuJoCo 3.3.2**，**LeRobot 0.6.1**；硬件 **RTX 2060 6 GB**（无真机）。
- **PyTorch+AMP**：Spatial/Object **70.0% / 88.0%**，p99 **1181 ms**。
- **Requested-FP16 / INT8 ONNX（width 16）**：p99 **601 / 532 ms**，Spatial 跌至 **41% / 40%**，Object 仍 **89%**。
- **图审计**：requested-FP16 与 INT8 导出为 **字节级相同 FP32 图** — INT8 行 **不是** 算子级量化。

**对 wiki 的映射：** [paper-smolvla-onnx-libero](../../wiki/entities/paper-smolvla-onnx-libero.md)

### 摘录 2：语言宽度消融

- 静态语言宽度 **16 / 24 / 32** tokens → Spatial **41% / 75% / 71%**；width 24 ONNX Spatial 与 PyTorch 基线相当（Wilson 区间重叠，χ² p=0.53），延迟约减半。
- 默认 width 16 会截断 **5/10** 条 Spatial 指令 — 部署评估必须报告 **接口约束 + 闭环成功率**，不能只看 bench 延迟。

**对 wiki 的映射：** 同上

### 摘录 3：配对验证（300 episode/suite）

- Spatial **56.7% → 33.0%**；Object **73.3% → 74.0%** — 换运行时后任务难度响应不同。

**对 wiki 的映射：** 同上（结论）

## 当前提炼状态

- [x] GitHub README + results 核查（2026-09-16）
- [x] wiki 映射：`wiki/entities/paper-smolvla-onnx-libero.md`
