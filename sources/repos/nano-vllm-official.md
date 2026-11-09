# Nano-vLLM｜面向学习的极简 LLM 推理引擎：官方资料与源码核验

- **类型：** repo（官方项目/代码/贡献指南）
- **项目：** https://github.com/GeeeekExplorer/nano-vllm
- **入库日期：** 2026-11-09
- **版本：** GitHub main 分支 2026-11-09 查阅；具体 commit SHA 未钉定，后续更新需复核
- **代码：** https://github.com/GeeeekExplorer/nano-vllm（公开；不是所有模块均能在单卡运行）
- **核对范围：** README.md（概述、安装、运行、benchmark）；example.py / bench.py（完整入口）；nanovllm/engine/llm_engine.py / scheduler.py（生成调度与后处理）；nanovllm/engine/model_runner.py（已阅读初始化、KV cache 和调用入口）。本次核对这些官方入口、模块和文档，不代表通读整个仓库所有源码。
- **通读：** 已完成
- **原创视频：** 用户只提供了视频推荐要点，原视频 URL 和 Discord 组队原帖仍未核实，不把视频观点写作项目官方承诺

## 一手资料

- [README.md](https://github.com/GeeeekExplorer/nano-vllm/blob/main/README.md)
- [example.py](https://github.com/GeeeekExplorer/nano-vllm/blob/main/example.py)
- [bench.py](https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py)
- [nanovllm/engine/llm_engine.py](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/llm_engine.py)
- [nanovllm/engine/scheduler.py](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/scheduler.py)
- [nanovllm/engine/model_runner.py](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/model_runner.py)

## 机制与复现要点

`LLM.generate` 先把 prompt 编码为 `Sequence`，`Scheduler` 在 waiting/running 队列间组织 prefill 和 decode 并调用 `BlockManager` 管理 cache；`ModelRunner.run` 负责前向与采样，`postprocess` 判断 EOS 或输出长度后释放块。

**官方使用入口：** `example.py` 中 `from nanovllm import LLM, SamplingParams` → `LLM(...).generate(...)`；`bench.py` 用输出 token 数 / 运行时间计算离线吞吐。

**贡献验证：** 先复现 Qwen3-0.6B `example.py`，再选 `Scheduler` / prefix caching 的边界条件编写正确性用例；固定同一 GPU、prompt 集、种子与生成长度，报告 tokens/s 和 P95 延迟（如有在线封装）。

**证据边界：** README 标称约 1200 行 Python；官方在 RTX 4070 Laptop 8GB、Qwen3-0.6B、256 请求的离线测例报告 1434.13 tokens/s，对照 vLLM 1361.84 tokens/s。这不是本站实测，也不代表生产线上更快。

## 对 Wiki 的映射

- [Nano-vLLM｜面向学习的极简 LLM 推理引擎](../../wiki/entities/nano-vllm.md)
- [LLM 开源贡献阶段路线](../../wiki/roadmaps/llm-inference-open-source-contribution-by-stage.md)
