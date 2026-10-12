# AgentGarten 项目页

> 来源归档（site / 官方项目页）

- **标题：** AgentGarten: Code Worlds for Evolving Agents
- **类型：** site
- **链接：** <https://mirros-lab.github.io/agent-garten/>
- **论文：** <https://arxiv.org/abs/2610.12374>
- **代码：** <https://github.com/MirroS-Lab/AgentGarten>
- **权重：** [AgentGarten-renderer](https://huggingface.co/MirroS-Lab/AgentGarten-renderer)
- **官方博客：** <https://mirros.ai/blog/worlds-for-evolving-agents>
- **入库日期：** 2026-10-12
- **沉淀到 wiki：** [`wiki/entities/paper-agentgarten.md`](../../wiki/entities/paper-agentgarten.md)

## 项目页摘要

- 世界的状态与交互规则由 simulator / game engine 执行，world 将几何条件导出到共享的神经渲染器；智能体通过生成画面交互，并把一轮经验写入 playbook 供后续轮次继承。
- 项目展示 hide-and-seek 的 shelter 与 ramp 行为，也展示 companion dog、one-lane bridge、sheep herding、quarry loader 等 code worlds。
- 项目页强调真实时间交互；吞吐的精确条件和边界以技术报告 §4.3 的 H100 稳态 benchmark 为准。

## 源码开放核查（2026-10-12）

- **已发布：** 官方仓库提供 neural renderer 训练/推理代码和渲染器 checkpoint。
- **待发布：** README TODO 仍将实时 rendering server / interactive frontend 与 code worlds / agent practice loop 标为未完成。完整的环境交互与 playbook 实验不能仅凭当前仓库直接复现。
- **许可证：** 仓库声明 Apache-2.0。

## 对 wiki 的映射

- [`wiki/entities/paper-agentgarten.md`](../../wiki/entities/paper-agentgarten.md)
- [`sources/papers/agentgarten_arxiv_2610_12374.md`](../papers/agentgarten_arxiv_2610_12374.md)
- [`sources/repos/agentgarten.md`](../repos/agentgarten.md)
- [`sources/blogs/mirros-agentgarten.md`](../blogs/mirros-agentgarten.md)
