# AgentGarten 官方仓库

> 来源归档（repo）

- **标题：** AgentGarten: Code Worlds for Evolving Agents
- **类型：** repo
- **链接：** <https://github.com/MirroS-Lab/AgentGarten>
- **论文：** <https://arxiv.org/abs/2610.12374>
- **项目页：** <https://mirros-lab.github.io/agent-garten/>
- **许可证：** Apache-2.0（README / `pyproject.toml`）
- **入库日期：** 2026-10-12
- **一句话说明：** 已开放 Cosmos3-Nano 几何条件神经渲染器的训练配方、checkpoint 和块式流式推理；完整 code-world 环境与 practice/playbook loop 尚未发布。
- **沉淀到 wiki：** [`wiki/entities/paper-agentgarten.md`](../../wiki/entities/paper-agentgarten.md)

## 开源边界（README，核查日 2026-10-12）

| 状态 | 内容 |
|---|---|
| **已开源** | `wm/` neural renderer 训练与推理；`scripts/run.sh` 训练入口；`Cosmos3Stream` 块式流推理；`prepare_serving` 手写 kernel / CUDA Graph serving；HF renderer checkpoint |
| **待发布** | streaming server / interactive frontend；code worlds 与 agents 的 rounds-of-play / review / playbooks practice loop |
| **运行要求** | Python 3.11+、PyTorch 2.9+、CUDA；按 README 安装 `pip install -e '.[data,media,wandb,te]'`，另下载 Cosmos3-Nano 与 Wan 2.2 VAE 权重 |

## README 中的渲染推理骨架

1. `inspect_artifact` / `load_artifact` 载入 student renderer。
2. `prepare_serving` 预热 serving kernels 与 CUDA graphs。
3. `Cosmos3Stream.start` 用 clean first-frame latent 和首块条件启动。
4. 每块通过少数 `RCM_ENDPOINTS` 去噪步骤生成 clean latent，再 `commit` 到有界 KV cache。
5. 下一个条件块沿相同接口进入；视频解码器输出 RGB observations。

首次使用会发生 kernel JIT 与 graph capture，应先 warm shapes。此代码覆盖 renderer，不等同于论文完整的 agent/world runtime。

## 交叉来源

- [`wiki/entities/paper-agentgarten.md`](../../wiki/entities/paper-agentgarten.md)
- [`sources/papers/agentgarten_arxiv_2610_12374.md`](../papers/agentgarten_arxiv_2610_12374.md)
- [`sources/sites/agentgarten-project.md`](../sites/agentgarten-project.md)
- [`sources/blogs/mirros-agentgarten.md`](../blogs/mirros-agentgarten.md)
