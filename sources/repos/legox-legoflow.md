# LegoFlow（LegoX/LegoFlow）官方代码仓库

- **类型：** repo / coding-agent data engineering
- **仓库：** <https://github.com/LegoX/LegoFlow>
- **默认分支：** master
- **源码状态：** 已公开
- **许可证：** Apache-2.0（GitHub repository metadata）
- **官方文档：** <https://legoflow-docs.legox.net/docs>
- **官方博客：** <https://www.legox.net/blog/legoflow/>
- **数据集：** <https://huggingface.co/datasets/Lego-X/LegoFlow-SWE>
- **核验日期：** 2026-10-08
- **一句话说明：** 面向 coding agent 的模块化数据工程框架，以 Curator 收集和验证 SWE 任务、Tracer 生成/筛选轨迹、Trainer 训练模型、Evaluator 运行 benchmark。

## 仓库结构与职责（README）

| Block | 主要职责 |
|-------|----------|
| Root | 将目标分解为 block workflow，并协调任务、资源和反馈 |
| Curator | 从 GitHub PR 等来源构建并验证 Harbor SWE tasks |
| Tracer | 在隔离任务环境中调用 coding-agent scaffold，记录、评分并格式化轨迹 |
| Trainer | 轨迹数据转换和模型 SFT；README 示例采用 LLaMA-Factory 格式 |
| Evaluator | 对 checkpoint 运行 coding-agent benchmark，发布评测报告 |

四个子 block 通过 agent plugin skills 调用；仓库采用递归子模块克隆：

~~~bash
git clone --recurse-submodules https://github.com/LegoX/LegoFlow LegoFlow
~~~

## 运行条件摘要

- Claude Code 或 Codex CLI 作为 agent 调用入口；Curator、Tracer、Evaluator 需要 OpenAI-compatible LLM endpoint。
- Curator 收集 GitHub PR 需要通过 GITHUB_TOKENS 提供 token。
- 任务构建和 rollout 使用 Docker 隔离；只有自行训练或托管 checkpoint 才需要 GPU。
- README 报告已在单节点 8×H800 80GB 上验证训练配置，多节点训练尚未接通。
- 官方博客称 Lego-RL 正在接入 Trainer；应视为路线中的集成工作，不写成稳定现成功能。

## 对 wiki 的映射

- 统一项目节点：[LegoFlow](../../wiki/entities/legoflow.md)
- 官方项目入口：[legoflow-project.md](../sites/legoflow-project.md)
- 数据集归档：[legoflow-swe.md](../datasets/legoflow-swe.md)
- 官方博客归档：[legoflow_2026-09-18.md](../blogs/legoflow_2026-09-18.md)
