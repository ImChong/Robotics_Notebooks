# AgenticGenTAMP — 作者项目页归档

- **类型：** project site / generalized TAMP / coding-agent program synthesis
- **正式论文：** Coding Agents for Generalized Task and Motion Planning Problems
- **网站：** https://agenticgentamp.github.io/
- **论文：** https://arxiv.org/abs/2609.30233
- **源码：** https://github.com/tomsilver/robocode
- **数据：** https://huggingface.co/datasets/prpl-group/agentic-gentamp-runs
- **核对日期：** 2026-11-10
- **核对范围：** 阅读作者项目页的 Generalized task and motion planning、Experimental setup/Access restrictions/Held-out evaluation、Gallery/Results/Effect of source access、Computation at test time、Citation 等公开内容，并核对其 Paper/Code/Dataset 指向。
- **通读：** 已完成
- **开源判断：** **已开源**，有可运行研究代码、官方运行指南和已公布实验运行数据；论文保留评测 seed 由作者按需提供，不能把公开数据误当成完全可得的隐藏种子。

## 核心证据

- 站点明确将同一环境内的“不同物体数量、配置、几何”称为不同 problem instances，**状态/动作空间、转移模型、奖励函数保持不变**；跨仿真环境通用迁移未被验证。
- Main setting：无网络/宿主机访问，无环境源码，只有 Python/NumPy/SciPy、reset/step/render，无预制 TAMP 组件。
- 训练阶段自由选择实验，每合成 run 模型预算 20 美元；结果冻结后，每环境 100 条同分布保留实例、每实例 60 秒限时，无测试时 LLM 调用。
- 研究材料链接到 25 个 KinDER 环境和 3 个 PDDLStream 域，并发布 policy rollout、研究代码及 Hugging Face runs。
- 网页部分动态统计需要 JS；精确数值以作者论文 Tables I–III 为准，避免误用静态占位符。

## 对应知识库

- [论文全文归档](../papers/agenticgenplan_coding_agents_generalized_tamp_arxiv_2609_30233.md)
- [RoboCode 源码阅读归档](../repos/tomsilver-robocode-agenticgenplan.md)
- [AgenticGenPlan 唯一项目实体](../../wiki/entities/paper-agenticgenplan-generalized-tamp-coding-agents.md)
