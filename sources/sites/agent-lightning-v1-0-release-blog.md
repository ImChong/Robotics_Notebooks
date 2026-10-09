# Agent Lightning v1.0：Harnessed Agentic RL 发布说明

- **类型：** Microsoft Research 官方博客
- **发布日期：** 2026-10-07
- **链接：** <https://www.microsoft.com/en-us/research/blog/agent-lightning-v1-0-a-3500-line-lightweight-agentic-rl-framework-for-training-agents-with-real-harnesses/>
- **项目代码：** <https://github.com/microsoft/agent-lightning>
- **技术报告：** <https://arxiv.org/abs/2608.17528>
- **官方文档：** <https://microsoft.github.io/agent-lightning/stable/>
- **开源状态：** MIT 许可；提供框架实现、示例和训练流程，训练仍需 verl / vLLM 与 GPU 运行栈。
- **一句话说明：** 微软研究院介绍 v1.0 的 Harnessed Agentic RL：让部署时的真实 agent harness 原样参与 RL rollout，通过 API Gateway 代理采集请求轨迹，并以 Trainer、Gateway、Rollout Controller 组成轻量控制面。

## 发布说明要点

1. **真实 harness 直接训练：** 工具协议、上下文管理、环境交互和 agent 控制流保留在真实 harness 内；将原模型 API 端点改指向 OpenAI 兼容 Gateway，即可记录模型调用并接入训练。
2. **约 3,500 行控制面：** API Gateway 记录 prompt、response、token 与 log probabilities；Rollout Controller 以本地进程或 Kubernetes Job 运行 agent；Customized Trainer 基于 verl 汇总样本并更新模型。
3. **从请求对到训练样本的四项挑战：** harness 的多轮请求可能导致 retokenization / sample merging 问题；同一 rollout 会切成不等数量样本，因而需正确处理 rollout-level advantage、rollout-level loss normalization，以及按可变工作量调度固定训练资源。
4. **共置异步训练：** rollout 与模型更新共用一组 GPU。系统收集到足够完成的 rollout 后，Gateway 暂停新请求、排空在途请求，执行模型更新后再恢复推理；未完成 rollout 组结转到后续步骤。发布说明报告实验中相较同步 RL 约 2 倍端到端加速、且使用的 GPU 少于传统分离式异步 RL；这是其测得结果，不代表任意环境的固定收益。
5. **Coding Agent 结果：** 基于 SWE-smith、mini-SWE-agent 和 Qwen3.5-9B 的完整流程，约 6,000 个训练样本将 SWE-bench Verified Pass@1 从 41.8% 提升至 56.4%（绝对提升 14.6 个百分点）。

## 文档入口

- [Quick Start](https://microsoft.github.io/agent-lightning/stable/01-quick-start/) — 单机本地 Calc-X 训练示例；文档说明策略推理和 verl 更新仍需要 GPU 栈。
- [Basics](https://microsoft.github.io/agent-lightning/stable/05-basics/) — Gateway、Rollout Controller 与 Customized Trainer 的职责，以及 rollout 与训练样本的区别。
- [Trainer Configuration](https://microsoft.github.io/agent-lightning/stable/20-trainer-configuration/) — 数据输入、trajectory / transition 聚合、rollout 级 advantage 与损失归一化配置。
- [Asynchronous Training](https://microsoft.github.io/agent-lightning/stable/35-asynchronous-training/) — 异步 rollout 组结转、Gateway drain、监控和 stale rollout 修正。
- [Coding Agent](https://microsoft.github.io/agent-lightning/stable/75-example-coding-agent/) — SWE-smith 编码 Agent 的数据、Kubernetes 与训练运行方式。

## 对 wiki 的映射

- 实体页：[wiki/entities/agent-lightning.md](../../wiki/entities/agent-lightning.md)
- 代码归档：[sources/repos/agent_lightning.md](../repos/agent_lightning.md)
- 技术报告：[sources/papers/agent_lightning_v1_technical_report.md](../papers/agent_lightning_v1_technical_report.md)
