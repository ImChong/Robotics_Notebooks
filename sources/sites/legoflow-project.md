# LegoFlow 官方项目文档与发布入口

- **标题：** LegoFlow: Easy and Interactive Code Data Engineering
- **类型：** project site / official docs
- **官方文档：** <https://legoflow-docs.legox.net/docs>
- **官方博客：** <https://www.legox.net/blog/legoflow/>
- **代码：** <https://github.com/LegoX/LegoFlow>（Apache-2.0）
- **公开数据：** <https://huggingface.co/datasets/Lego-X/LegoFlow-SWE>
- **核验日期：** 2026-10-08

## 项目入口核查

- 文档和博客将 LegoFlow 定义为面向 coding-agent 的代码数据工程流水线，覆盖仓库/PR发现、任务校验、轨迹生成、训练与评测。
- 主流程由 Root 协调 Curator、Tracer、Trainer、Evaluator 四个 block；每块有对应 skill、代码、依赖、脚本和 artifacts。
- 官方发布了 LegoFlow-SWE 数据集；细节见 [dataset source](../datasets/legoflow-swe.md)。
- 代码入口及默认分支、许可证见 [GitHub 仓库归档](../repos/legox-legoflow.md)。

## 开源状态

**项目代码：已开源（Apache-2.0）；任务与轨迹：已在 Hugging Face 公开。** HF 页面当前可见任务目录与约 9,767 条轨迹。数据集许可需与代码仓库许可分开核实；本次浏览器呈现的 dataset card 未明确展示单独许可证字段。

## 对 wiki 的映射

- [LegoFlow 项目实体](../../wiki/entities/legoflow.md)（代码、数据集与项目博客共用一个节点）
- [LegoFlow 官方博客归档](../blogs/legoflow_2026-09-18.md)
