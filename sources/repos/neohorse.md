# NeoHorse（TokenRhythm/NeoHorse）

> 来源归档

- **标题：** NeoHorse — agent-native open-weight models（NeoHorse-1 / NeoHorse-Jev）
- **类型：** repo
- **来源：** TokenRhythm Technologies / NeoHorse Team
- **链接：** <https://github.com/TokenRhythm/NeoHorse>
- **论文 / 报告：** <https://arxiv.org/abs/2609.08183>（仓内 [`TechnicalReport_NeoHorse_v1.pdf`](https://github.com/TokenRhythm/NeoHorse/blob/main/TechnicalReport_NeoHorse_v1.pdf)）
- **项目页：** <https://tokenrhythm.ai/>
- **许可：** Apache-2.0（NeoHorse-1）；NeoHorse-Jev 含 [Kev](https://github.com/jaredpalmer/kev) 衍生组件，见 `jev/README.md`
- **入库日期：** 2026-10-02
- **一句话说明：** **NeoHorse-1** 4B/9B 文本 agent 模型权重 + vLLM/SGLang 部署说明与 chat/tool 示例；**NeoHorse-Jev-4B** 结构化决策（Choice/Noul/Score）；**不含** routing harness 训练管线源码。
- **沉淀到 wiki：** [`wiki/entities/paper-neohorse-1.md`](../../wiki/entities/paper-neohorse-1.md)

---

## 仓库结构（截至 2026-10-02）

| 路径 | 说明 |
|------|------|
| `examples/chat.py` | OpenAI 兼容 chat 调用示例 |
| `examples/tool_call.py` | 工具调用示例 |
| `jev/` | NeoHorse-Jev-4B 决策模型文档与资源 |
| `assets/` | 评测结果图 |
| `TechnicalReport_NeoHorse_v1.pdf` | 与 arXiv 同步的技术报告 |

## 推理入口（README）

| 步骤 | 命令 / 说明 |
|------|-------------|
| 依赖 | `pip install -U vllm` 或 SGLang；`pip install requests` |
| 起服务（4B 示例） | `vllm serve /path/to/NeoHorse-1-4B --served-model-name neohorse-1-4B --max-model-len 262144 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder` |
| Chat | `python examples/chat.py --url http://127.0.0.1:8000 --model neohorse-1-4B` |
| Tool call | `python examples/tool_call.py --url http://127.0.0.1:8000 --model neohorse-1-4B` |

## 开源边界

- **已开源 / 已发布：** 模型 checkpoint（Hugging Face / ModelScope / GGUF / MLX）、本仓库推理示例与技术报告 PDF、NeoHorse-Jev 推理包。
- **未开源：** 论文所述 routing harness 在线系统、轨迹采集与六维语义质检流水线、SFT/OPD 训练实现与私有 harness 数据（仅方法描述 + 已发布权重）。
