# ApxInf（Rust 端侧推理引擎核心）

> 来源归档

- **标题：** ApxInf — Agentic Edge Inference Engine
- **类型：** repo
- **组织：** 无问芯穹（Infinigence AI）
- **代码：** <https://github.com/infinigence/ApxInf>
- **消费方：** [`RLinf/APXinf-robo`](apxinf-robo.md)（git submodule `apxinf/`）
- **入库日期：** 2026-09-16
- **一句话说明：** **Rust 实现、零外部依赖** 的端侧推理引擎核心：具身 AI 优先，定制 CUDA 算子（融合/CUTLASS/cuBLASLt），Agent 时代 model-port / kernel 工作流；由 APXinf-robo 以 Python binding（`apxinf_py`）暴露。
- **步骤 2.5：** **已开源** — 文档含 porting workflow、adding-a-new-model、FP8 calibration、onestep warm-start 等。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [APXInf 实体](../../wiki/entities/apxinf.md) | 引擎 vs Robo 封装的分层说明 |
| [APXinf-robo](apxinf-robo.md) | 机器人 preset、LIBERO eval、OpenPI serve 层 |
