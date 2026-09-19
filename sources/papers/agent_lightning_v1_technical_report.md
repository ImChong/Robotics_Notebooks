# Agent Lightning v1.0: Towards Harnessed Agentic RL

- **类型：** paper / technical report
- **链接：** <https://arxiv.org/abs/2608.17528>
- **PDF：** <https://arxiv.org/pdf/2608.17528>
- **机构：** Microsoft
- **代码：** <https://github.com/microsoft/agent-lightning>
- **入库日期：** 2026-09-19
- **一句话说明：** v1.0 技术报告：提出 **harnessed agentic RL** — 通过 API Gateway 代理在 **不改 agent harness** 的前提下采集轨迹，用轻量 Trainer / Controller / Gateway 三组件对接 verl 训练。

## 核心摘录（README / 公开摘要）

1. **设计原则：** 约 3,500 行代码；simplicity first；agent 经 v1.0 proxy 与模型交互，工具、上下文、控制流与环境保持 in the loop。
2. **部署形态：** 原生 Kubernetes Job rollout；亦支持本地 Controller。
3. **Coding Agent 结果：** 6K 样本 Qwen3.5-9B，SWE-bench Verified 41.8% → 56.4%。
4. **与 v0 关系：** v1.0 完全重构；初版论文 arXiv:2508.03680 描述早期「零代码改动 RL」思路。

## 对 wiki 的映射

- 实体页：[wiki/entities/agent-lightning.md](../../wiki/entities/agent-lightning.md)
- 仓库归档：[sources/repos/agent_lightning.md](../repos/agent_lightning.md)
