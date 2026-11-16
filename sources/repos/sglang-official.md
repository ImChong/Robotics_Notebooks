# SGLang｜大规模语言与多模态模型推理服务框架：官方资料与源码核验

- **类型：** repo（官方项目/代码/贡献指南）
- **项目：** https://github.com/sgl-project/sglang
- **入库日期：** 2026-11-09
- **版本：** GitHub main 分支 2026-11-09 查阅；具体 commit SHA 未钉定，后续更新需复核
- **代码：** https://github.com/sgl-project/sglang（公开；不是所有模块均能在单卡运行）
- **核对范围：** README.md（项目、快速开始、硬件支持、开发贡献）；python/sglang/srt/managers/tokenizer_manager.py（模块职责/入口）；python/sglang/srt/managers/scheduler.py（调度职责/核心类与模型执行依赖）。本次核对这些官方入口、模块和文档，不代表通读整个仓库所有源码。
- **通读：** 已完成
- **原创视频：** 用户只提供了视频推荐要点，原视频 URL 和 Discord 组队原帖仍未核实，不把视频观点写作项目官方承诺

## 一手资料

- [README.md](https://github.com/sgl-project/sglang/blob/main/README.md)
- [python/sglang/srt/managers/tokenizer_manager.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py)
- [python/sglang/srt/managers/scheduler.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)

## 机制与复现要点

SGLang 的 `python/sglang/srt/managers/tokenizer_manager.py` 在请求入口负责 tokenize/任务管理，`managers/scheduler.py` 的 Scheduler 管理 GPU worker 和批处理；模型执行层与 KV cache 等子系统集成。该生产框架与教学版是独立仓库。

**官方使用入口：** 官方 README 给出 `docker pull lmsysorg/sglang:latest` 与文档 quickstart；源码 `python/sglang/srt/managers/tokenizer_manager.py` 和 `scheduler.py` 明确 TokenizerManager / Scheduler 角色。

**贡献验证：** 先从 README 的 Development and Contributing、dev Docker 镜像、editable `python` 安装、Issue/PR 和回归测试进入；只选一个可证明失败的 case，附测试和可复现的前后性能。

**证据边界：** 支持多类硬件但具体 CUDA、TPU、NPU 支持随版本/模型而不同。Mini-SGLang 的约 5000 行代码不能视为 SGLang 全量实现。

## 对 Wiki 的映射

- [SGLang｜大规模语言与多模态模型推理服务框架](../../wiki/entities/sglang.md)
