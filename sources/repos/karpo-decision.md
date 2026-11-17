# Karpo Decision：官方推理代码与 System One 服务（源码归档）

- **标题：** karpo-decision — Inference code for the Karpo decision model
- **类型：** repo / 推理与评测适配器
- **开发方：** MachinePulse（MachinePulse Pte. Ltd.）
- **代码：** https://github.com/machinepulse-ai/karpo-decision
- **权重：** https://huggingface.co/collections/MachinePulseAI/karpo-decision
- **许可证：** 推理代码 Apache-2.0；模型权重须以独立 Hugging Face LICENSE 核对
- **核查版本：** 上游 main，2026-11-17 读取
- **核对范围：** 通读官方 README 的模型表、Install、Use、Serving、优化路径、System One 兼容性、Decision Index、Layout、Tests、License；阅读 `src/karpo_decision/decider.py` 的 `plan/decide/_run/_packs/logits` 实现、`src/karpo_decision/serve/server.py` 的 `Handler.do_POST/main`、`src/karpo_decision/serve/engine.py` 的 `DecisionIndexEngine`，核对 `pyproject.toml` 依赖及运行入口。没有逐一追踪所有注意力 kernel、训练实现或本地运行 GPU。
- **通读：** 已完成
- **开放状态：** 已开源可运行推理代码、HTTP server、Decision Index 适配器与测试；训练代码、训练集完整公开情况未在本仓得到确认。
- **沉淀到 wiki：** [Karpo Decision](../../wiki/entities/karpo-decision.md)

## 实际源码调用路径

1. `Decider.from_pretrained(directory, device="cuda")`：通过 `checkpoint.load` 取得处理器、骨干模型、配置和 readout；`Encoder` 为请求构造模板与 token/image 计划。
2. `Decider.decide(request)`：调用 `plan` 将状态、问题、选项编码；超长请求按问题拆分，每片保留完整状态；`_run` 合并请求并按 `token_budget` 分 pack。
3. `Decider.logits`：`pack(sequences, backbone, device)` → `self.model(...)` → `self.readout(hidden)` → `per_question(..., counts)`；`answer` 按类型和温度缩放后的 softmax 返回各字段概率。
4. HTTP 服务通过 `torchrun -m karpo_decision.serve.server` 启动，rank 0 提供 `POST /v1/systemone` / `GET /healthz`；`Handler.do_POST` 解析输入并检验 schema、图像及长度，再交给 `Ranks.run` / `Runner` 执行。
5. CLI 评测通过 `python -m decision_index run --engine karpo_decision.serve.engine:DecisionIndexEngine` 加载相同的 `Decider`，输出 Decision Index 的候选概率结果；评测工具需单独安装。

## 运行入口

```bash
pip install "karpo-decision @ git+https://github.com/machinepulse-ai/karpo-decision"
# Qwen3.5 系列按 README 使用 [qwen] extra
# 下载权重后
torchrun --standalone --nproc_per_node 1 -m karpo_decision.serve.server --checkpoint <checkpoint-dir> --port 18440
```

- 默认侦听 `127.0.0.1`，源码明确**没有内置身份认证与 TLS**；对外部署应自行加认证/网关和传输保护。
- 默认单序列长度 32,768；过长被拒绝而非默默截断。CUDA Graph / FP8 / Triton kernel 是可选路径，有对应 checkpoint 与硬件限制；模型卡数字不能作为所有硬件通用延迟。
- `pyproject.toml` 指定 Python >=3.12、PyTorch >=2.14 和 Transformers >=5.17；`qwen` extra 包括 `flash-linear-attention`。
- 当前仓的 README 仍可能沿用旧的模型权重 Apache-2.0 描述，**与模型仓更新授权文本可能有版本差**；商业使用须直接检查权重仓的最新 LICENSE。

## 来源与交叉映射

- [README（推理与部署）](https://github.com/machinepulse-ai/karpo-decision/blob/main/README.md)
- [Decider 核心](https://github.com/machinepulse-ai/karpo-decision/blob/main/src/karpo_decision/decider.py)
- [HTTP 服务](https://github.com/machinepulse-ai/karpo-decision/blob/main/src/karpo_decision/serve/server.py)
- [评测引擎](https://github.com/machinepulse-ai/karpo-decision/blob/main/src/karpo_decision/serve/engine.py)
- [依赖与许可元数据](https://github.com/machinepulse-ai/karpo-decision/blob/main/pyproject.toml)
- [权重与模型卡](../models/karpo-decision-weights.md)
- [Karpo Decision 知识页](../../wiki/entities/karpo-decision.md)
