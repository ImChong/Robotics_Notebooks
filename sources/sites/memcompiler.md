# MemCompiler 官方项目页

> 来源归档

- **标题：** MemCompiler: Compile, Don't Inject — State-Conditioned Memory for Embodied Agents
- **类型：** site（项目页）
- **链接：** <https://air-embodied-brain.github.io/MemCompiler/>
- **论文：** [arXiv:2605.07594](../papers/memcompiler_arxiv_2605_07594.md)
- **代码：** [air-embodied-brain/MemCompiler](../repos/air-embodied-brain-memcompiler.md)
- **模型：** [Hugging Face 模型条目](https://huggingface.co/xinrui-23/thor-5.0-16-Qwen2.5-14B-Instruct-Qwen2.5-7B-Instruct)
- **入库日期：** 2026-10-11
- **一句话说明：** 清华 AIR 等提出的状态条件记忆编译方法；每步以 Brief State 选择相关历史，并通过文本和 Soft-Mem 通道为执行器提供指导。

## 开源核查（2026-10-11）

| 项 | 状态 |
|----|------|
| 论文 | arXiv v2，CC BY 4.0 |
| 代码 | **已公开**：官方 GitHub 提供 ALFWorld / ScienceWorld 运行脚本、Manager–Executor 任务入口和训练模型评测入口 |
| 数据 | 论文声明 code and data available；README 给出环境数据目录和路径配置要求，具体许可与完整性以仓库、基准官方发布为准 |
| 权重 | 项目页提供 Hugging Face 模型链接；链接页面当前无 model card，权重说明与许可待核实 |
| 复现条件 | 依赖 ALFWorld / ScienceWorld、配置文件、LLM API 或本地模型及对应路径 |

## 项目页摘录

- 将 AMMI 一次性静态注入替换为逐步 SCMC；每步决策包括 EXPERIENCE、BRIEF、HYBRID、NOACTION。
- 两条输出通道：文本指导与 Soft-Mem 潜在 token；Brief State 在各步更新并反馈给后续编译。
- 页面展示 EmbodiedBench、ALFWorld / ScienceWorld 结果及 attention analysis；量化设置以论文表格为准。

## 对 wiki 的映射

- [MemCompiler 实体页](../../wiki/entities/paper-memcompiler-state-conditioned-memory.md)
- [arXiv 论文摘录](../papers/memcompiler_arxiv_2605_07594.md)
- [官方代码仓库归档](../repos/air-embodied-brain-memcompiler.md)
