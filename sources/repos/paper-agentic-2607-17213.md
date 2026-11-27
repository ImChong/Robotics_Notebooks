# openretriever/retriever — Retriever 官方开源仓库档案

- **GitHub 地址：** https://github.com/openretriever/retriever
- **默认分支：** `main`
- **开源协议：** Apache-2.0 License
- **关联论文：** *Retriever: Composing the Perception-Reasoning-Action Loop for Long-Horizon Manipulation*（[arXiv:2607.17213](https://arxiv.org/abs/2607.17213)）
- **关联项目页与文档：** [openretriever.org](https://openretriever.org) · [文档站 retriever.build](https://retriever.build) · [应用示例 GoldenRetriever](https://golden.retriever.build)（配套仓库 `openretriever/golden-retriever`）
- **核对边界：** 静态代码核对，未本地运行（基于仓库元数据、`README.md`、`files_sample` 与论文 Appendix D/F 核查，未在本地安装 `retriever-core` 或连接真机/相机执行）。

## 核心入口与目录结构（已核实）

- **包发布与 CLI 入口（`README.md`）：**
  - PyPI 分发包名 `retriever-core`（Python 3.11+，`pip install retriever-core`；导入名为 `retriever`，导出 `Flow`、`Pipeline`、`Rate`、`Trigger`、`Latest`、`io` 等核心抽象）。
  - 命令行工具 `retriever`：提供 `retriever demo webcam --visualize rerun`（拉取 `openretriever/webcam-demo` Hub pack 并通过 Rerun 可视化 `Camera -> ColorDetector -> Rerun` 闭环）、`retriever install`（源码检出后基于 Pixi 初始化环境）、`retriever run test`、`retriever run p0-release-readiness`、`retriever run public-surface-check`。
  - 图构建、调试与部署 API（`README.md` 与论文 Appendix D）：
    - 图定义：`with Pipeline(...)` 配合 `flow @ Rate(hz=...)` / `@Trigger(...)` 与 `.then(..., sync=Latest())`；
    - 编译与持久化：`pipe.build_ir()` / `ir.save("agent.ir.json")` / `IR.visualize()` / `IR.compile()`；
    - 执行与调试：`pipe.step(dt=0.1)` / `pipe.close_stepper()`（进程内单步调试），`pipe.run(backend="multiprocessing" | "dora", duration=...)`（多进程或 Dora 后端异步部署）；
    - 录制与重放：`pipe.record("session.rrd", ...)` / `pipe.replay(source, path="session.rrd")`（支持 MCAP / Rerun 轨迹记录与精确重放）。
- **已核查文档与教程目录（`files_sample`）：**
  - 核心文档：`docs/API.md`、`docs/architecture.md`、`docs/concepts_lineage.md`、`docs/guide_flow.md`、`docs/guide_runtime.md`、`docs/guide_execution.md`、`docs/guide_temporal.md`、`docs/hub.md`、`docs/quickstart.md`、`docs/examples.md`。
  - 工程指南与类型规范：`docs/guides/debugging.md`、`docs/guides/advanced_zero_copy.md`、`docs/guides/data_eventstream_v1.md`、`docs/guides/data_spec_isolation_v1.md`、`docs/guides/flow_typing_standard.md`、`docs/guides/perception_types_v1.md`、`docs/guides/spatial_types_v1.md`、`docs/guides/language_types_v1.md`、`docs/guides/symbolic_types_v1.md`、`docs/guides/type_composition_v1.md`。
  - 讲义与交互式 Notebook：`docs/tutorials/lectures/l01_retriever_mental_model.md` 至 `l10_stateful_flows_and_reset_contracts.md`；`docs-site/public/notebooks/flow-fundamentals.ipynb`、`standard-types.ipynb`、`step-debug-replay.ipynb`、`time-and-sync.ipynb`、`using-the-hub.ipynb`。
- **生态边界（`README.md`）：**
  - `openretriever/retriever`（`retriever-core`）仅聚焦核心编程模型、IR 与运行时；
  - `openretriever/golden-retriever` 承载应用机器人示例与 Hub pack（感知、记忆、语言规划、仿真/可视化通道等）；
  - Retriever Hub 提供无需克隆全仓即可按名称与固定 commit 解析加载 Flow/类型/Pipeline 的清单协议。

## 复现与运行依赖边界

- **运行环境：** Python 3.11+；源码开发与测试依赖 Pixi 任务包装（`retriever install` / `retriever run test`）。
- **可选后端与可视化：** 若启用 Dora 分布式传输后端，论文 Appendix F 注明使用 `dora-rs >= 0.3.12`；实时可视化与 webcam 示例依赖 `rerun-sdk`（`pip install "retriever-core[demo]" rerun-sdk`）。
- **外部模型与真机限制：** 核心运行时本身不绑定特定大模型或真机驱动，但复现论文 Section V–VI 中的 Retriever-0 长程操作管线需要外部 VLM 服务（如 Gemini）、微调后的 $\pi_{0.5}$ 或扩散技能策略权重以及双臂 PiperX / Meta Quest 3 硬件接口（示例层归于 GoldenRetriever）。

## 与 wiki 的映射

- 对应实体页：[Retriever: Composing the Perception-Reasoning-Action Loop for Long-Horizon Manipulation](../../wiki/entities/paper-agentic-2607-17213.md)
