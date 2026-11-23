# Tencent WePocketModel — 官方推理源码核查

- **类型：** repo / 官方代码
- **代码仓库：** https://github.com/Tencent/WePocketModel
- **关联论文：** [WePM arXiv:2611.27365](../papers/wepm_hierarchical_conflict_reconciliation_arxiv_2611_27365.md)
- **模型合集：** [WePocketModel](../models/wepocketmodel-collection.md)
- **许可：** 仓库代码 Apache-2.0；模型权重许可须查看各 HF 模型页面
- **核对日期：** 2026-11-23
- **核对范围：** 官方 README 英文（概述、性能、Inference/Quick start/Hardware/License）；逐文件阅读 examples/gpu/{generation,embedding,decision}/run.sh、inference/infer.py、inference/runtime.py 的入口/推理逻辑、inference/checkpoint.py 的 runtime_sources 选择、inference/README.md 的 CLI 参数及示例。

## 开放判断

**已公开推理代码及模型权重，论文训练流程仅部分可复现**。README 标出 CUDA / CPU Transformers 与 CPU GGUF，且存在可实际调用的 G/E/J 推理入口；未核实到论文约 1 亿样本、MGVB/ULKD/SEMO 多任务训练所需训练脚本、任务数据与复现实验配方，不将开源推理等价为端到端论文训练开源。

## 真实推理入口（与 wiki Mermaid 时序图对齐）

1. `bash
   bash examples/gpu/generation/run.sh single_text
   bash examples/gpu/embedding/run.sh
   bash examples/gpu/decision/run.sh
   `
2. 三个脚本分别调用 `inference/infer.py`，指定 `--task generate / embed / decision` 及 `--content-json`（生成/嵌入）或 `--decision-json`（决策）。命令写 JSON 结果文件。
3. `inference/infer.py` 通过 `model_source.resolve_model_path` 解析模型；含 `bundle.json` 时选择 GGUF CPU 分支，否则实例化 `runtime.UniModelTransformers`，执行 `generate(content, max_new_tokens)` / `embed(content, dims)` / `decide(request, images)`。
4. `UniModelTransformers` 加载 Transformers 的 `Qwen3_5ForConditionalGeneration` 与 auxiliary/task modules；运行时 `_prepare_inputs` 处理有序图文、调用视觉编码及 MGVB 压缩，嵌入读取 embedding slot 与归一化 head；决策读取 decision slot 与 direct decision head；生成沿语言模型自回归解码。
5. CPU GGUF 走 `inference/infer_cpu.py` 和 `gguf_runtime.UniModelGGUF`；依赖 `inference/build_gguf.py` 编译 native 后端。**这与 GPU Transformers 路径不同**。

## 复现注意

- README 建议 4B BF16 GPU 至少 16 GiB VRAM；其约 8.5 GiB 是权重指引（不含推理峰值）；CPU FP32 约 17 GiB 权重、建议 24 GiB RAM；GGUF Q4_K_M 3.19 GiB 是磁盘文件量而非运行 RAM。
- `--trust-model-code` 会执行所选模型快照 `runtime_sources/` 的 Python，需审查代码并固定 revision，默认使用仓库自带 runtime。
- 具体成功与性能未经本次实际运行验证；仓库内容核对不构成端上部署测试。

## 对 wiki 的映射

- [WePM 论文独立实体及源码时序图](../../wiki/entities/paper-wepm-hierarchical-conflict-reconciliation.md)
