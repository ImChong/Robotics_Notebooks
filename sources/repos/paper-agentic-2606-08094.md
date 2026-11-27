# VinRobotics/vla.cpp 官方代码仓库核查

- **仓库地址：** https://github.com/VinRobotics/vla.cpp
- **许可证 / 默认分支：** Apache-2.0 / `main`（记录时 277 stars）
- **关联论文与归档：** [arXiv:2606.08094](https://arxiv.org/abs/2606.08094) / [论文原文核对归档](../papers/paper-agentic-2606-08094.md)
- **核对边界：** 静态代码与文档核对（基于仓库 README、目录样本与发布说明，未在本地编译或运行真机）。

## 仓库结构与可运行入口

根据仓库根目录与 `README.md` / `eval/` 清单核查，项目包含完整的 C++ 构建、模型打包、服务端/CLI 推理及仿真/真机客户端脚本：

- **构建与核心目录：** `CMakeLists.txt`、`Dockerfile`、`src/`、`include/`、`cmake/`（构建时由 CMake 拉取固定 tag `b11223` 的 `ggml-org/llama.cpp`）。构建产物为 `vla-cli`、`vla-server`、`vla-bench` 与 `libvla`。
- **Python 绑定：** `bindings/python/pyproject.toml`、`bindings/python/vla_cpp/__init__.py`、`bindings/python/vla_cpp/_ffi.py`。
- **文档与后端支持：** `docs/ARCHITECTURE.md`、`docs/USAGE.md`、`docs/MODELS.md`、`docs/QUANTIZATION.md`、`docs/EVAL.md`、`docs/ADOPTION.md`、`docs/backend/`（CUDA、ROCm、SYCL、OpenVINO、Metal、Hexagon）及 `docs/benchmark/`。
- **单机推理入口：** `./build/vla-cli -hf vrfai/smolvla-libero-gguf --image assets/front.jpg --text "pick up the black bowl" --pretty`（或通过 `--ckpt` 指定本地 `.gguf` 文件）。
- **服务化与评测客户端：** `./build/vla-server <model>.gguf`（默认绑定 ZeroMQ `tcp://*:5555`）；评测脚本位于 `eval/`（如 `eval/run_libero_server.sh`、`eval/run_libero_client.sh`、`eval/client/vla_cpp_client.py`、`eval/client/adapters.py`、`eval/client/run_libero_eval.py`、`eval/client/run_simpler_eval.py`、`eval/client/run_ALOHA_client_direct.py`、`eval/docker-compose.yml`）。
- **配套仓库（外链）：** `khanhnd61-vr/ggml-compiler`（PyTorch 到 ggml C++ 移植与验证）与 `khanhnd61-vr/lerobot`（`vla-simd` 分支提供 `lerobot-vla-cpp` 真机客户端）。

## 开源状态判定

- **判定结果：** 已开源（包含 C++ 运行时源码、ZeroMQ/Protobuf 服务端、Python FFI 绑定、多后端构建说明与仿真/真机评测客户端；预训练 GGUF 权重托管于 Hugging Face `vrfai` 组织下）。
