# GroundingPI 官方代码仓库

> 来源归档

- **标题：** GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives — Official Code
- **类型：** repo / grounding / visual-foundation-model
- **代码：** <https://github.com/groundingpi/GroundingPI>
- **项目页：** <https://groundingpi.github.io/>
- **论文：** <https://arxiv.org/abs/2609.39601>
- **模型权重：** <https://huggingface.co/GroundingPI/GroundingPI>
- **Hugging Face 论文页：** <https://huggingface.co/papers/2609.39601>
- **检查日期：** 2026-11-09
- **一句话说明：** 官方仓库包含 GroundingPI 的模型实现、训练与评测配置、vLLM/Transformers 服务、客户端示例和批量图像标注工作流。
- **沉淀到 wiki：** [GroundingPI 论文实体页](../../wiki/entities/paper-groundingpi.md)

## 仓库能力与复现入口

- **推理服务：** `run.py` 提供环境准备和服务启动入口；默认指南使用 vLLM Transformers backend，也提供原生 Transformers 参考服务。服务暴露 OpenAI-compatible 图文 API。
- **客户端：** `grounding_pi/` 提供 HTTP 客户端与可视化；`examples/` 包含单图预测和 JSONL 批处理示例。
- **评测：** `eval/` 与 `configs/eval/` 覆盖论文报告的 34 项 grounding benchmark；默认 recipe 为 smoke test，完整套件按 Evaluation Guide 配置运行。
- **训练：** `train/` 与 `configs/train/`、`configs/release/` 提供训练流程与配置。README 指出完整输入缓存、manifest 和相符的运行时需由使用者准备；默认训练指南面向对应的 PPU 环境。
- **模型结构：** MoonViT-V2（Kimi K3）视觉编码器、多模态投影器和 Qwen3-4B 语言骨干。
- **依赖与环境：** README 的 serving workflow 目标为 Linux x86_64 / Python 3.12；需先准备兼容 accelerator 的 Torch 与 vLLM 0.18.x 运行时，再安装项目依赖。
- **开放边界：** 仓库提供 grounding VLM checkpoint 的推理/训练/评测代码和配置；下游策略对比需要 StarVLA / OpenWAM 各自环境及策略 checkpoint，不等同于公开了直接执行机器人动作的通用策略。

## 已逐文件核对的真实源码入口（2026-11-09）

- `run.py`：解析 `setup serve/train/eval`，决定 `configs/release/vlm_vllm_gpu.yaml` 或指定配置，再通过 `.venv-serve/bin/python scripts/run.py ...` 执行。
- `scripts/run.py`：配置 `entrypoint: vlm-vllm` → `python -m infer.serve_vllm`；启动前验证树结构与 checkpoint，组装受控环境。
- `infer/serve_vllm.py`：验证 Transformers 5.7.0 项目 fork / vLLM 0.18.x、准备 `infer/overlay/prepare_model_overlay.py` 服务适配层，再 exec `vllm.entrypoints.openai.api_server`；单请求只允许一张图，BF16/eager。
- `examples/predict.py`：真实 CLI 参数 `--image --phrase --task {bbox,point}`，调用 `GroundingPi.predict`，把 `result.json` 写入输出目录，解析有效才生成 `prediction.png`。
- `grounding_pi/client.py`：`build_refer_bbox_prompt` / `build_refer_point_prompt` → base64 图像 + 文字构造 `/chat/completions` 请求，关闭特种 token 过滤与 token 间插空格；`parse_response` 检查 `finish_reason == "stop"`，拒绝截断结果。
- `grounding_pi/protocol.py`：定义严格 GAM 语法和 `parse_answer`，`<0>`–`<999>` 坐标 token，非法或不完整输出不得伪装为成功。
- `docs/VLLM.md`、`docs/INFERENCE.md`、`vla/README.md`：确认服务环境、单图推理、独立动作 backbone 对照的差异；`vla/` 并未提供 GroundingPI grounding checkpoint 的直接即插即用 action adapter。

上述为**源码及静态文档阅读**，未实际运行服务、评测、训练、GPU 或真机测试。

## 许可提示

仓库原始贡献声明 Apache 2.0。组合权重及依赖还包含 Kimi K3 和其他第三方材料，其许可条件仍然适用；使用及再分发前应阅读根目录 `LICENSE` 与 `THIRD_PARTY_NOTICES.md`。

## 对 wiki 的映射

- 主实体页：[wiki/entities/paper-groundingpi.md](../../wiki/entities/paper-groundingpi.md)
- 项目主页归档：[sources/sites/groundingpi-github-io.md](../sites/groundingpi-github-io.md)
- 论文摘要归档：[sources/papers/groundingpi_arxiv_2609_39601.md](../papers/groundingpi_arxiv_2609_39601.md)
