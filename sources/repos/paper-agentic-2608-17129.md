# vineet2104/MG-VQA

- **仓库地址：** https://github.com/vineet2104/MG-VQA
- **默认分支：** `main`
- **开源协议：** MIT License（仓内第三方模型权重与资产遵循各自上游协议，包括 Meta DTC CC BY-NC 4.0、MegaPose、BOP、ambientCG CC0 1.0、GraspGen、SAM3、Ultralytics）
- **对应实体：** [MG-VQA: Manipulation Grounded Visual Question Answering with VLMs](../../wiki/entities/paper-agentic-2608-17129.md)
- **核对边界：** 静态代码与文档核对（基于已抓取仓库文件列表、README 预览及上游关联仓库元信息，未在本地 GPU/Blender 环境执行仿真运行）

## 仓库定位与验证结论

该仓库为论文《MG-VQA: Manipulation Grounded Visual Question Answering with VLMs》（arXiv:2608.17129）的官方开源实现，完整包含 **MG-VQA-Sim** 仿真评测环境、**MG-VQA-Bench**（600 题）下载与重放脚本、VLM 多模态推理闭环以及 PyBullet + Blender-EEVEE + GraspGen/SAM3 感知操作栈。

- **开源完整度：** 核心评测与仿真代码、Prompt 模板及 600 题基准测试入口已开源；仓库 README「Next updates」注明 **MG-VQA-Hard** 难题集、更多操作工具与训练数据生成引擎尚未包含在当前版本中，计划后续发布。
- **数据集与资产托管：**
  - 基准快照与问题：`https://huggingface.co/datasets/bhatvineet/mg-vqa-bench`（下载脚本钉定 revision `78c4d8adf7a5c68060bc32b340c8a746933f659c`）
  - 319 个真实扫描物体资产池：`https://huggingface.co/datasets/vineet-datasets/agentic-mqa-objects`

## 已核对目录与关键入口

- **启动与 CLI 入口：**
  - `run_mgvqa.sh`：Shell 启动器，支持多模式批量执行（`direct,perception,agentic`）与 `--workers` / `--gpus` / `--recycle` / `--dry-run`。
  - `mgvqa/__main__.py` & `mgvqa/cli.py`：Python CLI 入口（`python -m mgvqa`）。
  - `mgvqa/run_benchmark.py`：冻结场景库重放评测主入口。
- **核心编排与推理闭环：**
  - `mgvqa/orchestrator/parallel.py` & `mgvqa/orchestrator/summary.py`：多进程并行 worker 调度、进程回收（推荐每 4 个 episode 回收一次）与结果汇总。
  - `mgvqa/worker.py` & `mgvqa/worker_entry.py`：单回合生命周期管理、场景恢复（`mgvqa/scene_snapshot.py`）与答题评分（`mgvqa/metrics.py`、`mgvqa/answer.py`）。
  - `mgvqa/reasoning_loop.py` & `mgvqa/singlepass.py`：分别实现 Perception/Manipulation 多轮工具闭环与 Direct 单轮推理。
  - `mgvqa/tool_registry.py`、`mgvqa/prompts.py`、`mgvqa/prompt_templates/`：工具注册、模式权限限制、匿名工具反馈格式化及三套系统/用户 Prompt 模板。
- **模型适配、感知与操作后端：**
  - `models/catalog.py`、`models/dispatch.py`、`models/protocols.py`、`models/providers/`（anthropic/gemini/local/nvidia/openai_compat/openrouter/xai）：多厂商 VLM API 路由与工具调用协议适配。
  - `perception/`、`manipulation/grasp.py`、`manipulation/push.py`、`manipulation/primitives.py`：场景图/SAM3 点分割、`grasp` 与 `push` 原语实现，GraspGen 配置位于 `checkpoints/graspgen/graspgen_robotiq_2f_140.yml`。
  - `owg_robot/` & `robo_blender/` & `config/pyb/env.yaml`：PyBullet UR5 + Robotiq 2F-140 仿真环境、网格资源及 Blender-EEVEE 渲染同步。
- **安装、检查与报告脚本：**
  - `scripts/finish_setup.sh`、`scripts/download_weights.sh`、`scripts/download_assets.py`、`scripts/download_benchmark.py`、`scripts/preflight.py`、`scripts/build_report.py`。
- **外部运行依赖：** Linux、NVIDIA GPU + 匹配 `nvcc` CUDA 工具链、Conda（`environment.yml`）、Blender 5.1、SAM3 与 GraspGen 权重及 VLM API Key（`NV_API_KEY` 或原生提供商 Key）。
