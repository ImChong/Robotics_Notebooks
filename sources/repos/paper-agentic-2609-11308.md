# 2AM / AgentMem 官方仓库核对归档（utomm/AgentMem-Action-Model）

- **仓库地址：** https://github.com/utomm/AgentMem-Action-Model
- **关联论文：** [2AM（AgentMem）论文来源归档](../papers/paper-agentic-2609-11308.md)（arXiv:2609.11308）
- **默认分支：** `main`
- **许可证：** `CC-BY-4.0`（仓库自主编写代码使用 CC BY 4.0，第三方组件遵循 `NOTICE` 与 `licenses/` 下的 `LICENSE_GEMMA.txt`、`LICENSE_OPENPI.txt`）
- **开源状态：** **已开源**（公开 VLM Agent 训练、Action Model 流匹配推理服务、资源下载工具、4 基准评测客户端及 Hugging Face 模型权重与标注集合；仓库随项目发展以 `AgentMem` / `agent-mem-act` 命名）
- **核对边界：** 静态代码结构、文件样本与 README 核对，未本地运行

## 仓库目录与关键入口

基于 `utomm/AgentMem-Action-Model` 仓库 `main` 分支根目录、README 与目录树样本核对：

1. **包配置与依赖清单**
   - `pyproject.toml`、`requirements/data.txt`、`requirements/inference.txt`、`agentmem/agent/requirements.txt`、`agentmem/agent/requirements-inference.txt`、`agentmem/action/requirements.txt`。
   - 数据、训练、Agent 推理与 Action 推理支持分离虚拟环境（`.venv-train`、`.venv-agent`、`.venv-action`、`.venv-robomme` 等）。

2. **资源下载与统一评测入口**
   - `agentmem/assets.py`：一键拉取基准模型与标注，示例命令 `python -m agentmem.assets --benchmarks robomme --components base vlm vla`，配置清单位于 `configs/agentmem/huggingface_assets.json`。
   - `agentmem/eval.py`：统一本地评测入口（支持 `robomme`、`rmbench`、`robodojo`、`libero_mem`），通过多环境参数（`--vlm-python`、`--vla-python`、`--env-python`）及 `--vlm-device` / `--vla-device` 调度双模型闭环，配置位于 `configs/eval/*.json`。

3. **Agent 侧训练与在线服务（`agentmem/agent/`）**
   - 训练入口：`agentmem/agent/train.sh`（默认以 `NPROC_PER_NODE=8` 启动 `configs/agentmem/robomme.json`，基于 `Qwen3-VL-4B-Instruct` 进行 3 epoch LoRA 微调）、`agentmem/agent/train_dense.py`、`agentmem/agent/deepspeed_zero2.json`。
   - 数据准备与视频数据集：`agentmem/agent/prepare.py`、`agentmem/agent/prepare_robomme.py`、`agentmem/agent/export_annotations.py`、`agentmem/agent/video_dataset/build.py`、`agentmem/agent/video_dataset/validate.py`、`agentmem/data/robomme.py`。
   - 在线推理与流式记忆维护：`agentmem/agent/serve.py`、`agentmem/agent/streaming.py`、`agentmem/agent/robodojo_streaming.py`、`agentmem/agent/minimal_client.py`、`agentmem/agent/overlay_reader.py`。

4. **Action Model 侧推理服务（`agentmem/action/`）**
   - 策略与模型实现：`agentmem/action/model.py`、`agentmem/action/modules.py`、`agentmem/action/policy.py`、`agentmem/action/action_chunk.py`、`agentmem/action/pose.py`。
   - 服务化与执行：`agentmem/action/serve.py`、`agentmem/action/run.py`、`agentmem/action/benchmark_policy.py`、`agentmem/action/robomme.py`。

5. **四基准评测客户端与仿真环境（`agentmem/evaluation/`、`environments/`）**
   - `agentmem/evaluation/libero_mem/`（`client.py`、`online.py`）
   - `agentmem/evaluation/robomme/`（`agent_rollout.py`、`streaming.py`、`task_context.py`、`memer_client.py`、`stopcube_oracle.py`）
   - `agentmem/evaluation/rmbench/`（`vlm_client.py`、`oracle_client.py`、`diagnostics.py`、`recovery.py`）
   - `agentmem/evaluation/robodojo/`（`client.py`）
   - `environments/`：通过 `.gitmodules` 挂载四个仿真器分支子模块（版本记录于 `environments/versions.json`）。

6. **配套项目主页源码仓（仅展示站点）**
   - `https://github.com/agent-mem-action-model/agent-mem-action-model.github.io` 为纯静态 HTML/CSS/JS 项目主页源码（含历史 `/workshop/` 归档与 2026-11 AgentMem 主页资源），不含模型训练与策略代码。

## 对 wiki 的映射

- [2AM（AgentMem）实体页](../../wiki/entities/paper-agentic-2609-11308.md)
