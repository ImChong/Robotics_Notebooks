# zhengzihaoPKU/RoboECC

- **仓库地址：** <https://github.com/zhengzihaoPKU/RoboECC>
- **许可证 / 默认分支：** MIT License / `master`（截至核对快照 116 stars）
- **对应实体：** [RoboECC: Multi-Factor-Aware Edge-Cloud Collaborative Deployment for VLA Models](../../wiki/entities/paper-agentic-2603-20711.md)
- **核对边界：** 静态代码与文档核对（基于仓库文件清单、README 文档及入口说明），未在双端 GPU 环境本地运行。

## 仓库定位与开源范围（Step 2.5 核查）

- **开源状态：** **部分开源**（核心规划与运行时控制层已开源，外部 VLA 模型执行后端与实测带宽数据未内置）。
- **已开源组件：**
  - `pyproject.toml`：定义 Python >= 3.10 包结构与 CLI 入口（`roboecc-search`、`roboecc-train-bandwidth`），可选 `[lstm,dev]` 依赖组引入 PyTorch。
  - `src/roboecc/profiles.py`：框架无关的模型层（`ModelProfile`、`LayerProfile`）、硬件（`HardwareProfile`）与网络（`NetworkProfile`）配置 schema。
  - `src/roboecc/segmentation.py`：离线部署切分搜索（`search_optimal_split`），按 Roofline 模型 `max(compute time, memory time)` 与网络传输耗时搜索最优云边切分点 `cut`（云端前缀阶段数）。
  - `src/roboecc/bandwidth_lstm.py`、`src/roboecc/train_bandwidth.py`：单层 many-to-one LSTM 带宽预测器（默认窗口 20、隐层维度 32，支持 `--csv` 时序训练与 `--synthetic-samples` 合成冒烟测试）。
  - `src/roboecc/pool.py`、`src/roboecc/network.py`、`src/roboecc/runtime.py`：参数共享池（`ParameterSharingPool`）、双阈值策略、原子化本地切分切换与运行时控制。
  - `src/roboecc/adapters.py`：OpenVLA 块级 `reconfigure` RPC 适配桥，以及受约束的 CogACT 参数映射适配器。
  - `src/roboecc/llama_block.py` & `examples/llama_block_internal.py`：将单个 LLaMA Decoder Block 展开为 5 个规划阶段（`input RMSNorm -> self-attention -> post-attention RMSNorm -> MLP expand -> MLP down`）的代价建模抽象。
  - `src/roboecc/cli.py` & `examples/toy_scenario.json`：示例场景搜索入口。
  - `tests/`（`test_adapters.py`、`test_bandwidth_lstm.py`、`test_llama_block.py`、`test_network.py`、`test_pool_runtime.py`、`test_profiles.py`、`test_segmentation.py`）：34 个 CPU 单元与回归测试（无 PyTorch 时跳过 2 个 LSTM 测试）。
- **未包含 / 待外部对接部分（据官方 README 明确声明）：**
  - OpenVLA、CogACT、LIBERO、SimplerEnv 均为外部依赖，未在本仓库打包。
  - OpenVLA 适配器依赖外部提供真实跨设备切分执行后端；CogACT 适配器要求云前缀 DiT 顺序并预加载共享池参数（与旧版独立 CogACT 客户端的边前缀顺序不兼容）。
  - LLaMA 块内 5 阶段切分仅用于规划与传输量建模，不提供跨双 GPU 的算子级 `LlamaDecoderLayer.forward()` 替换。
  - 未发布 A100↔Jetson Thor/Orin 的实测带宽轨迹 CSV、官方预训练 LSTM 权重文件及双机联调回归脚本。

## 核对的 CLI 与 API 入口

1. **切分搜索入口：** `roboecc-search examples/toy_scenario.json`（调用 `src/roboecc/cli.py` → `src/roboecc/segmentation.py`）。
2. **带宽预测器训练入口：** `roboecc-train-bandwidth --csv traces/a100_thor_bandwidth.csv --column bandwidth_bytes_per_second --output checkpoints/bandwidth_lstm.pt`（或 `--synthetic-samples 300 --epochs 10` 做流水线冒烟测试）。
3. **测试入口：** `PYTHONPATH=src python -m unittest discover -s tests -v`。
