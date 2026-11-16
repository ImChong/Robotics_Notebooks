# Self-Refine — 官方项目主页

> 来源归档（官方站点）

- **站点：** <https://selfrefine.info/>
- **类型：** project / llm / self-feedback
- **论文：** [Self-Refine / arXiv:2303.17651v2](../papers/self_refine_arxiv_2303_17651.md)
- **代码：** <https://github.com/madaan/self-refine>
- **仓库归档：** [madaan/self-refine](../repos/self-refine-madaan.md)
- **收录日期：** 2026-11-11

## 项目页核实的开放状态

- **已开源。** 主页头部提供 Paper、Code 和 Demo 入口；Code 明确指向作者官方 GitHub madaan/self-refine。
- 公开的是不同文本生成任务的提示词、Python 实验脚本与相关数据；无需训练新模型，但复现需要能调用相应 LLM API。
- 主页给出 FEEDBACK 与 REFINE 的说明，以及单任务停止函数伪代码；代码实际行为须结合任务入口核实。

## 关键事实

- 一套 LLM 通过初稿、自反馈、改稿的提示循环优化输出，反馈强调「可执行」和「可定位」。
- 保留前几轮的输出及反馈历史，停止条件由任务定义。
- 项目演示并非机器人真机实验，不能直接宣称可提升实体控制可靠性。

## 对 wiki 的映射

- [Self-Refine 单一实体节点](../../wiki/entities/paper-self-refine-2303-17651.md)

## 参考来源

- <https://selfrefine.info/>
- <https://github.com/madaan/self-refine>
