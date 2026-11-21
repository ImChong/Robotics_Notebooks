# Being-H0 仓库核查

> 来源归档（repo / BeingBeyond GitHub）

- **URL：** <https://github.com/BeingBeyond/Being-H0>
- **机构 / 个人：** BeingBeyond（智在无界）
- **仓库核查commit：** `ae92e46`（188 个文件，含 69 个 `.py`/`.sh` 文件）
- **开源结论：** **部分开源（已开源推理/真机服务端与预训练/对齐权重；预训练与后训练主训练脚本标注 TODO）**。
- **核对范围：** 已通过 `gh api` 核对 `BeingBeyond/Being-H0 @ ae92e46` 完整文件树、`README.md`，并直接阅读 `beingvla/inference/vla_internvl_inference.py` 与 `beingvla/models/motion/m2m/aligner/run_server.py` 首部实现；未在真机或 GPU 环境下运行脚本。
- **通读：** 已完成

## 核心源码入口与模块

- **手部运动推理与渲染**：`beingvla/inference/vla_internvl_inference.py`（加载 `Being-H0-{1B,8B,14B}-2508` 与双路 `Being-H0-GRVQ-8K` wrist+finger 码本，经 MANO 前向解算与渲染）
- **真机策略服务端与评测**：`beingvla/models/motion/m2m/aligner/run_server.py`（`PolicyWrapper` 封装 `<IMG_CONTEXT>` / `<PROP_CONTEXT>` / `<ACTION_CHUNK_i>` 并通过 `InternVLInferenceServer` 对外服务）、`eval_policy.py`、`run_client_example.py`
- **运动分词器与物理对齐**：`beingvla/models/motion/m2m/tokenizer/quantizer/part_grvq.py`、`beingvla/models/motion/m2m/utils/mano_process.py`
- **生态演进**：后续交叉本体训练代码整合于 <https://github.com/BeingBeyond/Being-H>（Being-H0.5）。
