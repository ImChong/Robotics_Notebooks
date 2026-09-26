# laya-coreml（FluidInference/laya-coreml · Swift LayaManager 权重）

- **标题:** laya-coreml（FluidInference Hugging Face 发布）
- **链接:** https://huggingface.co/FluidInference/laya-coreml
- **类型:** repo / weights / coreml / system-one
- **维护:** FluidInference（独立转换；非 Convai 官方）
- **许可:** Apache-2.0（跟随上游 `convaiinnovations/laya` multilingual 权重）
- **上游权重 pin:** `convaiinnovations/laya` 子目录 `multilingual/` @ revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`
- **上游代码:** https://github.com/NandhaKishorM/laya
- **消费方:** [FluidInference/FluidUse](https://github.com/FluidInference/FluidUse) 的 `LayaManager`（macOS 14+）
- **转换与验证:** [FluidInference/mobius](https://github.com/FluidInference/mobius/tree/main/models/computer-use/laya/coreml) · `benchmark/suites.jsonl` 等
- **最后核查:** 2026-09-26
- **入库日期:** 2026-09-26

## 开源状态（步骤 2.5）

- **已开源：** HF 上完整 Core ML bucket（`.mlmodelc`）+ `tokenizer.json` + 校验元数据；随 `LayaManager.load()` 下载并 SHA-256 校验（FluidUse README）。
- **训练：** 无；322M mmBERT-base + decision head 权重与上游一致，仅运行时格式为 Core ML。

## 核心内容摘要

1. **多 bucket 策略：** `L128`–`L1024` × FP16（614 MB/桶，32 option slots）；`e8` 变体仅 int8 embedding table（约 30% 体积，精度与 fp16 差 ≤0.5 pt on full benchmark）。
2. **算力分配：** 短 prompt（L128）默认 **CPU + Neural Engine**；更长 bucket 在 M5 Pro 上 **All units（含 GPU）** 更低延迟（如 L512：9.0 ms vs ANE-only 27.5 ms）。
3. **延迟（M5 Pro，warm）：** L128 **3.6 ms**（CPU+ANE）；全 suite 中位 **5.2 ms**/问 vs PyTorch CPU **61.6 ms**（FluidUse Benchmarks）。
4. **parity：** 16 fixture 题 argmax 全一致；laya 公开 10 套应用 suite（3899 问）Swift Core ML 与 PyTorch reference **准确率逐套对齐**（见 HF model card 表）。
5. **输入格式：** `[CLS] question: [SEP] ([MASK])* [SEP] [SEP]`；`marker_map` 标记各 option 的 `[MASK]` 位置。

## 对 wiki 的映射

- **wiki/entities/laya-coreml.md** — Core ML 运行时（Swift 权重主路径）
- **wiki/entities/fluiduse.md** — `LayaManager` 与 CLI demo
- **wiki/entities/laya.md** — 上游 schema 与 checkpoint 对照
