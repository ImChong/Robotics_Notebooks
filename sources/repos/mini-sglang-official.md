# Mini-SGLang｜精简但完整的在线 LLM Serving 系统：官方资料与源码核验

- **类型：** repo（官方项目/代码/贡献指南）
- **项目：** https://github.com/sgl-project/mini-sglang
- **入库日期：** 2026-11-09
- **版本：** GitHub main 分支 2026-11-09 查阅；具体 commit SHA 未钉定，后续更新需复核
- **代码：** https://github.com/sgl-project/mini-sglang（公开；不是所有模块均能在单卡运行）
- **核对范围：** README.md（安装、在线服务、benchmark）；docs/structures.md（架构及消息流）；docs/features.md（后端、缓存、overlap scheduling）；python/minisgl/__main__.py / server/api_server.py / scheduler/scheduler.py / engine/engine.py（入口与模块级阅读）。本次核对这些官方入口、模块和文档，不代表通读整个仓库所有源码。
- **通读：** 已完成
- **原创视频：** 用户只提供了视频推荐要点，原视频 URL 和 Discord 组队原帖仍未核实，不把视频观点写作项目官方承诺

## 一手资料

- [README.md](https://github.com/sgl-project/mini-sglang/blob/main/README.md)
- [docs/structures.md](https://github.com/sgl-project/mini-sglang/blob/main/docs/structures.md)
- [docs/features.md](https://github.com/sgl-project/mini-sglang/blob/main/docs/features.md)
- [python/minisgl/__main__.py](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/__main__.py)
- [python/minisgl/server/api_server.py](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/server/api_server.py)
- [python/minisgl/scheduler/scheduler.py](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/scheduler/scheduler.py)
- [python/minisgl/engine/engine.py](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/engine/engine.py)

## 机制与复现要点

官方 `docs/structures.md` 描述用户请求由 API Server 交给 Tokenizer，Rank 0 Scheduler 接收 token IDs 并转发给各 TP rank 的 Scheduler，各自 Engine 推理后将输出交给 Detokenizer，再回 API。控制消息走 ZeroMQ，多 GPU 张量通信走 NCCL。

**官方使用入口：** `python -m minisgl --model 'Qwen/Qwen3-0.6B'` 进入 `python/minisgl/__main__.py` 的 `launch_server()`；源码 `server/api_server.py`、`scheduler/scheduler.py`、`engine/engine.py` 对应 API、批调度与模型执行。

**贡献验证：** 将 `--cache naive` 与 radix cache、`MINISGL_DISABLE_OVERLAP_SCHEDULING=1` 开关做单变量消融；校验文本/输出长度一致，再比较 TTFT、吞吐与延迟。

**证据边界：** 当前主要面向 Linux 和支持 CUDA 的平台；官方离线 benchmark 示例使用 H200、线上示例使用 4×H200；不应把这些图表数值直接推广到个人显卡。

## 对 Wiki 的映射

- [Mini-SGLang｜精简但完整的在线 LLM Serving 系统](../../wiki/entities/mini-sglang.md)
- [LLM 开源贡献阶段路线](../../wiki/roadmaps/llm-inference-open-source-contribution-by-stage.md)
