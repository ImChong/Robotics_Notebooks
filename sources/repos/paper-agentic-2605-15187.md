# articraftresearch/Articraft（Articraft 官方源码仓库归档）

- **仓库地址：** https://github.com/articraftresearch/Articraft
- **默认分支：** `main`
- **开源协议：** Apache-2.0（`LICENSE`）
- **仓库定位：** 论文《Articraft: An Agentic System for Scalable Articulated 3D Asset Generation》（arXiv:2605.15187）的官方受支持继任代码库（替代原原型仓库 `mattzh72/articraft`），实现文本/单图驱动的可动 3D 资产（URDF / 可摆姿 USDZ）智能体生成、校验、查看与物理仿真测试。
- **核对范围：** 基于官方仓库元数据快照（`context.json` 中 `README.md` 与 `files_sample` 目录采样）完成静态结构核对。
- **复现边界：** 静态代码核对，未本地运行（未执行 `uv run articraft` 或真实 LLM API 调用）。

## 仓库目录与入口核对

### 1. 运行入口与工具链
- **CLI 与应用入口**：通过 `uv sync --group dev` 安装依赖，在 `.env` 配置 API Key（支持 OpenAI 默认、Anthropic、Gemini、OpenRouter）后运行：
  - `uv run articraft "<prompt>"` / `uv run articraft --image reference.png "<prompt>"`（对应 `src/articraft/app.py`）
  - `uv run articraft view runs/<run-id>`：打开交互式资产版本与关节查看器
  - `uv run articraft simulate runs/<run-id>`（需 `uv sync --group sim`）：执行跌落测试、滑动摩擦与关节释放仿真校验
- **Python SDK API**：`src/articraft/api.py` 提供同步 `articraft.generate(...)` 与异步 `articraft.generate_async(...)` 接口，返回包含状态与产物路径（`result.status`、`result.artifact`）的结果对象，支持导出可摆姿 USDZ（`src/articraft/_usd.py`）。

### 2. 核心源码模块（`src/articraft/`）
- **Agent Harness 与编排层**（`src/articraft/agent/`）：
  - `harness.py`：维护编辑—编译—探针—修复主循环
  - `compaction.py` & `prompts/compaction.md`：长对话上下文压缩管理
  - `record.py` & `events.py`：记录轨迹、工具调用与成本元数据
  - `workspace/local.py`：隔离工作区，暴露单一可写脚本与只读 SDK 文档
  - `provider/{openai,anthropic,gemini,openrouter}.py`：多模型后端适配（仅 OpenAI/Anthropic/Gemini 支持参考图像）
- **Agent 工具集**（`src/articraft/agent/tools/`）：
  - `read.py`、`edit.py`、`write.py`、`compile.py`、`exec_command.py`、`write_stdin.py`、`view_image.py`、`images.py`
- **编译器与结构化反馈**（`src/articraft/compiler/`）：
  - `worker.py`、`_child_process.py`：独立进程执行模型构建与检测
  - `feedback.py`、`result.py`：汇总编译异常、QC 规则与自定义测试结果为结构化反馈
- **3D 建模 SDK**（`src/articraft/sdk/`）：
  - `assembly.py`、`bodies.py`：核心 `ArticulatedObject`、`Part` 与关节装配接口
  - `_collision.py`、`_overlap.py`：几何重叠与碰撞一致性校验
  - `_mesh/`（`boolean.py`、`core.py`、`health.py`、`profiles.py`、`refine.py`、`section_loft.py`、`shell_partition.py`、`sweeps.py`、`weld.py`）：网格布尔、放样、扫掠、壳体分割与网格健康检查
  - `ambientcg.py`：材质资源接入；`sdk/docs/build123d/`：内嵌 CAD 参考文档
- **基准与示例**：`benchmarks/sdk_benchmark.py`、`examples/hinged_box/main.py`、`examples/mesh_knob/main.py`

## 与知识库映射

- 对应实体页：[Articraft 实体页](../../wiki/entities/paper-agentic-2605-15187.md)
- 对应论文归档：[Articraft 论文原文笔记](../papers/paper-agentic-2605-15187.md)
