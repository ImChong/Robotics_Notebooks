# TensorRT-LLM｜NVIDIA 大模型推理运行时与 LLM API：官方资料与源码核验

- **类型：** repo（官方项目/代码/贡献指南）
- **项目：** https://github.com/NVIDIA/TensorRT-LLM
- **入库日期：** 2026-11-09
- **版本：** GitHub main 分支 2026-11-09 查阅；具体 commit SHA 未钉定，后续更新需复核
- **代码：** https://github.com/NVIDIA/TensorRT-LLM（公开；不是所有模块均能在单卡运行）
- **核对范围：** README.md（产品边界、文档、快速开始）；CONTRIBUTING.md（Issue、Review、CI、DCO、测试）；examples/llm-api/quickstart_example.py（官方 LLM API 可执行入口）；tensorrt_llm/llmapi/llm.py（LLM/GenerationExecutor 等接口依赖与输出类）。本次核对这些官方入口、模块和文档，不代表通读整个仓库所有源码。
- **通读：** 已完成
- **原创视频：** 用户只提供了视频推荐要点，原视频 URL 和 Discord 组队原帖仍未核实，不把视频观点写作项目官方承诺

## 一手资料

- [README.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/README.md)
- [CONTRIBUTING.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CONTRIBUTING.md)
- [examples/llm-api/quickstart_example.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llm-api/quickstart_example.py)
- [tensorrt_llm/llmapi/llm.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/llmapi/llm.py)

## 机制与复现要点

官方 `examples/llm-api/quickstart_example.py` 以 `from tensorrt_llm import LLM, SamplingParams` 创建模型、调用 `llm.generate`；`tensorrt_llm/llmapi/llm.py` 使用 GenerationExecutor、tokenizer / postprocessor 组件组织请求与输出。其贡献指南规定 Issue、review、CI、DCO 签署。

**官方使用入口：** 运行 `examples/llm-api/quickstart_example.py`：`LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0')` + `SamplingParams` + `llm.generate`；入口依赖 `tensorrt_llm/llmapi/llm.py` 内的 executor 组件。

**贡献验证：** 先按 `CONTRIBUTING.md` 创建 Issue；从单文件 accuracy test、API stability、performance regression 小改动开始，遵守项目 pre-commit 与 DCO sign-off，记录具体 GPU 与版本。

**证据边界：** README 标注 CUDA/PyTorch/发行版本会变化；部分模型与高性能路径依赖特定 NVIDIA GPU。TensorRT（通用网络 builder/runtime）和 TensorRT-LLM（LLM API/serving）不可当作同一技术页。

## 对 Wiki 的映射

- [TensorRT-LLM｜NVIDIA 大模型推理运行时与 LLM API](../../wiki/entities/tensorrt-llm.md)
- [LLM 开源贡献阶段路线](../../wiki/roadmaps/llm-inference-open-source-contribution-by-stage.md)
