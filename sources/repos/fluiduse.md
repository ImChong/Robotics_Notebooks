# FluidUse（FluidInference/FluidUse · Apple Silicon 本地计算机使用）

- **标题:** FluidUse
- **链接:** https://github.com/FluidInference/FluidUse
- **类型:** repo / computer-use / macos / accessibility / system-one
- **维护:** FluidInference
- **许可:** Apache-2.0（仓内 CUA-S1-FORMS 组件 MIT，源自 [Cua](https://github.com/trycua/cua)）
- **Swift Package:** `from: "0.3.0"` — https://github.com/FluidInference/FluidUse.git
- **项目展示（Laya Core ML Python 收据，非本仓）：** https://www.madewithlaya.com/builds/laya-coreml
- **Laya Core ML 权重:** https://huggingface.co/FluidInference/laya-coreml
- **表单专用模型:** https://huggingface.co/FluidInference/cua-s1-forms-coreml（706K，经 [FluidAudio](https://github.com/FluidInference/FluidAudio) 服务）
- **上游 Laya:** https://github.com/NandhaKishorM/laya · HF https://huggingface.co/convaiinnovations/laya · https://huggingface.co/convaiinnovations/laya-typed-decisions
- **最后核查:** 2026-09-26
- **入库日期:** 2026-09-26

## 开源状态（步骤 2.5）

- **已开源：** GitHub 完整 Swift 源码 + SPM 发布 + HF Core ML 资产按需下载；需 macOS **Accessibility** 授权（表单 demo）。
- **范围边界：** 模型在**给定 profile 实体**内做字段–值匹配；不读简历、不代写长文、默认不点 Submit（README Scope）。

## 核心内容摘要

1. **表单填充闭环：** `AccessibilityFormDriver` / `WebFormDriver` 快照可编辑字段 → `FormSchema.renderOptions` 构造 choice → **CUA-S1-FORMS** 或 profile 映射 → `driver.type` / `click`；Neural Engine 上约 **0.9 ms/决策**（与 PyTorch 24k 行合成测试精度一致）。
2. **Laya typed decisions：** `LayaManager` 加载 [laya-coreml](./laya-coreml-fluidinference.md) 的 **128 + 512** 默认 bucket + `tokenizer.json`；`answer(state:questions:)` 输出 `choice` / `score` / `noul`；M5 Pro 短问约 **3.7 ms**（README 相对 M1 Max GPU ~27 ms 上游数字）。
3. **CLI / Demo：** `FluidUseLaya answer|tetris|2048|benchmark`；SwiftUI `LayaTetrisDemo`、`GLiClass2048Demo`；`swift run -c release FluidUseDemo` 交互填表。
4. **GLiNER2Manager（同仓）：** 设备端分类头，HF [gliner2-5-base-coreml](https://huggingface.co/FluidInference/gliner2-5-base-coreml) / multilingual；与 Laya 并列的 **短文本分类** 路径，非本 ingest 主轴。
5. **文档:** [Benchmarks.md](https://github.com/FluidInference/FluidUse/blob/main/Benchmarks.md) · [DecisionModelSupport.md](https://github.com/FluidInference/FluidUse/blob/main/Documentation/DecisionModelSupport.md)

## 对 wiki 的映射

- **wiki/entities/fluiduse.md** — 计算机使用 + on-device 决策 harness
- **wiki/entities/laya-coreml.md** — Core ML Laya 权重与延迟
- **wiki/entities/laya.md** — 上游 Convai Laya 对照
