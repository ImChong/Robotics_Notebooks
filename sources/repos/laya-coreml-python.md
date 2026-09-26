# Laya-CoreML（mizorewww/laya-coreml · Python Core ML 运行时）

- **标题:** Laya-CoreML（Python）
- **链接:** https://github.com/mizorewww/laya-coreml
- **类型:** repo / decision-engine / system-one / coreml-runtime
- **作者:** mizorewww（社区独立端口；上游 Laya 为 Convai Innovations）
- **许可:** Apache-2.0
- **PyPI:** https://pypi.org/project/laya-coreml/（`pip install laya-coreml`）
- **展示页:** https://www.madewithlaya.com/builds/laya-coreml
- **HF 权重（示例）:**
  - https://huggingface.co/aac6fef/laya-multilingual-coreml-ane（Snake / 短 ANE L96）
  - 另有 CPU+GPU 512/1024 token 英文与 typed-decisions bundle（见 README「Available checkpoints」）
- **上游 Laya:** https://github.com/NandhaKishorM/laya · https://huggingface.co/convaiinnovations/laya
- **MLX  sibling:** [laya-mlx](./laya-mlx.md)
- **最后核查:** 2026-09-26
- **入库日期:** 2026-09-26

## 开源状态（步骤 2.5）

- **已开源：** GitHub 推理/转换/ benchmark 脚本 + PyPI + HF 多 bundle；**非 Convai / 非 Apple 官方**。
- **平台：** Apple Silicon · macOS 15+ · Python 3.11–3.13；推理 wheel 无 PyTorch 依赖（`[convert]` extra 另需 torch）。

## 核心内容摘要

1. **ANE 短决策：** M3 Max 单问 **4.98 ms P50**（ANE FP16 L96）；相对 compiled MLX FP16 **1.39× 速度、2.78× 系统能耗/决策**（见 `docs/ANE_BENCHMARKS.md`）。
2. **保真：** 三通用 FP16 checkpoint **189/189** 验证题与上游选中标签一致；ANE L96 **59/59** fitting 题通过 0.02 概率漂移门限。
3. **Demo：** `laya-coreml-snake` 完整游戏环 **~49–50 decisions/s**（600 步 episode，零死亡）。
4. **API：** `import laya_coreml as laya` → `laya.load(hf_id)` → `predict(state, questions_dict)`；ANE bundle **96 token 总上限**，更长上下文用 `aac6fef/laya-multilingual-coreml` 等 CPU+GPU 包。
5. **与 FluidInference 权重区别：** [FluidInference/laya-coreml](./laya-coreml-fluidinference.md) 为 **Swift `LayaManager` 多 bucket `.mlmodelc`**；本仓为 **Python `coremltools` 运行时 + 独立 HF 命名空间**，勿 interchange 路径。

## 对 wiki 的映射

- **wiki/entities/laya-coreml.md** — Core ML 实体（Python 路径）
- **sources/sites/madewithlaya-laya-coreml.md** — 收据与 demo 链接
- **wiki/entities/laya-mlx.md** — 同作者 MLX 端口对照
