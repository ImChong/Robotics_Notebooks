# Made with Laya — Laya-CoreML 展示页

> 来源归档（ingest）

- **标题：** Laya-CoreML: 4.98 ms decisions on the Neural Engine, 2.78x better energy than MLX
- **类型：** site / project-showcase
- **URL：** https://www.madewithlaya.com/builds/laya-coreml
- **入库日期：** 2026-09-26
- **作者（页内标注）：** mizorewww · @mizorewww
- **Python 实现：** https://github.com/mizorewww/laya-coreml · [PyPI `laya-coreml`](https://pypi.org/project/laya-coreml/)
- **HF 权重（页内 Snake / ANE 短上下文）：** https://huggingface.co/aac6fef/laya-multilingual-coreml-ane
- **上游 Laya：** https://github.com/NandhaKishorM/laya · https://huggingface.co/convaiinnovations/laya
- **Swift / FluidUse 路径（本页未主推，但同属 Core ML 生态）：** [FluidUse](../../sources/repos/fluiduse.md) · [FluidInference/laya-coreml](https://huggingface.co/FluidInference/laya-coreml)

## 一句话摘要

**Made with Laya** 上的 **Python Core ML 端口** 收据页：M3 Max 上 ANE FP16 单问 **P50 4.98 ms**，相对 compiled MLX FP16 约 **2.78× 更低系统能耗**；含可离线运行的 `laya-coreml-snake` 终端 demo 与公开 benchmark 数字。

## 公开信息要点（截至 2026-09-26）

- **延迟：** 4.98 ms P50 / 5.31 ms P95（M3 Max，ANE FP16，单问 91 token 垫至 96）
- **能耗：** 相对 MLX FP16，系统能耗/决策约 **2.78×** 改善（W8 palette 变体约 3.19×）
- **Demo：** `pip install 'laya-coreml[demo]'` → `hf download aac6fef/laya-multilingual-coreml-ane` → `laya-coreml-snake`
- **平台：** Apple Silicon · macOS 15+ · Python 3.11–3.13；推理无需 PyTorch / Transformers / MLX
- **类别：** Tools & apps（页内 Category）

## 开源状态（步骤 2.5）

- **已开源：** GitHub [mizorewww/laya-coreml](https://github.com/mizorewww/laya-coreml) + Apache-2.0 + PyPI + HF 多 bundle（含 ANE L96 与 CPU+GPU 长上下文变体）。
- **非 Convai 官方发布**；独立转换端口，与 [Laya-MLX](../repos/laya-mlx.md) 同作者生态。
- **与 FluidInference 分工：** [FluidInference/laya-coreml](https://huggingface.co/FluidInference/laya-coreml) 面向 **Swift `LayaManager`（FluidUse）** 的多 bucket 发布；Python 仓为 **独立 PyPI 运行时**，权重 ID 与打包策略不同，勿混为同一包。

## 对 wiki 的映射

- [`wiki/entities/laya-coreml.md`](../../wiki/entities/laya-coreml.md) — Core ML 运行时实体（Python + Swift 双栈）
- [`wiki/entities/fluiduse.md`](../../wiki/entities/fluiduse.md) — Swift 计算机使用 harness（`LayaManager`）
- [`wiki/entities/laya.md`](../../wiki/entities/laya.md) — 上游 typed decision 对照
