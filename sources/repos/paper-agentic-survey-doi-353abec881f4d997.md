# SCOPE 官方代码仓归档（HindsboNikolaj/SCOPE）

- **官方仓库：** https://github.com/HindsboNikolaj/SCOPE
- **关联论文：** [SCOPE: Real-Time Natural Language Camera Agent at the Edge（DOI: 10.1145/3757279.3785641 / arXiv:2606.02951）](../../wiki/entities/paper-agentic-survey-doi-353abec881f4d997.md)
- **开源协议：** MIT License（代码仓库 `LICENSE`；其中 `city-street` 等第三方 Blender 场景资产保留各自原许可协议，如 BlendSwap #30474 CC-BY）
- **默认分支与版本：** `main`（当前对应 Release `0.2.0`，2026 年 11 月更新；论文初版对应 `0.1.0`）
- **开源状态判定：** **已开源**（智能体主循环、OpenAI 兼容工具 Schema、Blender PTZ 工具与全景拼接、VLM 客户端适配、536 任务基准 v1/v2 CSV、评测与评委面板脚本、Dockerfile 及 GPU 并行评测器均已公开；4 个 Blender `.blend` 场景文件需通过 `scripts/03_download_scenes.sh` 从 [Hugging Face `HindsboNikolaj/scope-benchmark`](https://huggingface.co/datasets/HindsboNikolaj/scope-benchmark) 下载，真实硬件 AXIS PTZ 驱动未内置在 Blender 仿真工具模块中）
- **核对边界：** 静态代码核对，未本地运行（依据官方仓库目录树、`README.md`、`src/scope/agent/client.py`、`src/scope/tools/schema.json`、`src/scope/tools/blender_tools.py`、`src/scope/tools/vlm_clients.py`、`src/scope/eval/runner.py` 及 `scripts/run_eval_pipeline.sh` 核对模块划分与执行入口）

## 仓库结构与核心模块（基于实测代码与文件清单）

- **智能体主循环与思考模式控制（`src/scope/agent/`）：**
  - `src/scope/agent/client.py`：核心类 `AgentClient` 通过 OpenAI 兼容 Chat Completions + Tools API（环境变量 `AGENT_API_BASE`、`AGENT_MODEL_ID`、`AGENT_TOOL_CHOICE`）发送 `prompts/agent_system_prompt.md` 与可用预设列表，并在 `.ask(...)` 中循环执行工具调用直至模型输出最终文本回答或达到 `SCOPE_MAX_TURNS`（默认 40 轮，超限抛出 `TurnLimitExceeded`）。内置 `_strip_reasoning` / `_strip_channels` 清理 `<think>`、`<analysis>` 及 Harmony 通道标记。
  - `src/scope/agent/thinking.py`：识别模型族的思考模式（`ThinkingMode.TOGGLE` / `ALWAYS` / `NEVER`）并在每次请求通过 `chat_template_kwargs` 显式声明。
- **工具 Schema 与 Blender PTZ/VLM 执行层（`src/scope/tools/` & `src/scope/blender/`）：**
  - `src/scope/tools/schema.json`：定义 9 个 OpenAI Function Calling 工具，对应代码函数名为 `zoom_bounding`、`count_pointing`、`query_answer`、`take_image`、`home_action`、`get_presets`、`go_to_preset`、`track_object`、`ptz_adjust`。
  - `src/scope/tools/blender_tools.py`：在 Blender `bpy` 环境下实现上述 9 个工具（其中 `track_object` 在仿真中为 stub 占位）；感知工具通过 `_capture_for_view(view_type)` 支持 `current`（调用 `screenshot_camera_view` 截取当前相机视口）或 `full`（触发全景扫描拼接或读取全景缓存），并附带白屏/欠曝检查（`_is_flat`、`_is_dull`）。
  - `src/scope/tools/vlm_clients.py`：抽象基类 `VLMClient`（`caption`、`query`、`detect`、`point`）及工厂函数 `create_vlm_from_env()` / `create_vlm_from_config()`，适配 `MoondreamREST`、`MoondreamServer`、`MoondreamLocal`、`MoondreamPhoton`、`QwenVLServer`、`QwenVLLocal` 等多种本地或服务端 VLM 后端，并包含 `qwen_native_grounding`、`gemma_native_grounding`、`molmo2_points` 等坐标/点定位解析器。
  - `src/scope/blender/helper_funcs.py`、`preset_helpers.py`、`panorama_cache.py`、`agent_banner.py`、`presets_banner.py`：负责 3D 视口裁剪截图、相机预设管理、全景图缓存及 Blender 交互式 UI 横幅。
- **批量评测、评委与指标脚本（`src/scope/eval/` & `scripts/`）：**
  - `scripts/run_eval_pipeline.sh`：串行评测标准入口，按顺序执行：（1）带 GUI 启动 Blender 运行 `src/scope/eval/runner.py`（读取 `QUESTIONS_CSV`，逐题加载对应 `.blend` 场景并设置 `Material Preview` 着色，调用 `AgentClient.ask(...)` 记录结果与耗时至 `raw_results.csv`）；（2）调用 `scripts/11_judge.py` 评分；（3）调用 `scripts/12_metrics.py report` 生成统计报告。
  - `scripts/run_fast.py`：v0.2.0 新增多容器/多进程 GPU 并行评测入口，启动 6 个 Blender worker 常驻加载场景并使用硬件 OpenGL 离屏渲染，配合 `scripts/13_judge_panel.py` 执行三评委多数投票评测。
  - `scripts/01_install.sh` ~ `07_smoke_test.py`：依赖安装（含 Blender 内置 Python 依赖检查 `install_blender_deps.py`）、模型拉取、Hugging Face 场景下载、预设安装及冒烟测试。
- **基准数据、配置与容器化（`benchmark/`、`configs/`、`prompts/`、`docker/`）：**
  - `benchmark/scope_v2.csv`（默认 536 题）、`benchmark/scope_v2_smoke.csv`（24 题冒烟集）与 `benchmark/archive/scope_536_v1.csv`（论文原版 536 题）。
  - `configs/agent_config.yaml` 与 `configs/presets/*.yaml`：提供 Qwen3-4B/8B/30B-A3B/32B/Next-80B 与 Moondream2/3、Qwen2.5-VL-7B 等组合配置模板。
  - `docker/Dockerfile`、`docker/Dockerfile.hardware-gl`、`docker/build-gpu.sh`：无显示器 GPU 服务器渲染镜像。

## 文档与映射

- 仓库文档清单：`docs/architecture.md`、`docs/tool_reference.md`、`docs/CONFIGURATION.md`、`docs/SETUP.md`、`docs/RUN_FAST.md`、`docs/HEADLESS.md`、`docs/JUDGING.md`、`docs/PAPER_RESULTS.md`、`docs/RESULTS.md`、`docs/RESULTS_V2.md`、`docs/FULL_VIEW.md`。
- 对 wiki 的映射：[SCOPE](../../wiki/entities/paper-agentic-survey-doi-353abec881f4d997.md)
