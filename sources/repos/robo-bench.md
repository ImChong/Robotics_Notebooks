# RoboBench（官方实现与数据）

> 来源归档

- **标题：** RoboBench: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain
- **类型：** repo + benchmark toolkit + HF dataset
- **组织：** 北京大学 / BAAI 等（见论文）
- **代码：** <https://github.com/yulin-luo/RoboBench>
- **论文：** <https://arxiv.org/abs/2510.17801>
- **项目页：** <https://robo-bench.github.io>
- **Hugging Face 数据：** <https://huggingface.co/datasets/LeoFan01/RoboBench>
- **Hugging Face 结果：** <https://huggingface.co/datasets/lyl010221-pku/RoboBench-Results>
- **入库日期：** 2026-07-12
- **一句话说明：** 开源 **RoboBench 评测脚本**、**6092 QA 多模态数据集** 与 **18 模型 leaderboard 结果**；规划维含 **MLLM-as-world-simulator** 与 DAG 原子动作模板；依赖各 MLLM API 或本地权重（以 README 为准）。

## 仓库与数据入口（策展）

| 资源 | 链接 | 说明 |
|------|------|------|
| GitHub | <https://github.com/yulin-luo/RoboBench> | 评测代码与配置 |
| 数据集 | <https://huggingface.co/datasets/LeoFan01/RoboBench> | 五维 QA + 场景素材 |
| 结果 | <https://huggingface.co/datasets/lyl010221-pku/RoboBench-Results> | 官方 leaderboard 归档 |
| 项目页 | <https://robo-bench.github.io> | Demo、管线图、交互 leaderboard |

## 对 wiki 的映射

- 实体页：[RoboBench](../../wiki/entities/robo-bench.md) — 五维 taxonomy、规划评测框架、主要发现与复现入口。
- 交叉：[VLA](../../wiki/methods/vla.md)、[ESI-Bench](../../wiki/entities/esi-bench.md)、[EWMBench](../../wiki/entities/ewmbench.md)（同属具身评测生态，侧重点不同）。

## 2026-11-15 官方入口与实现复核

项目页实际链接 GitHub、HF 数据与结果。通读当前 README、PROMPTS_AND_PIPELINE 文档、CLI、数据/提示构建、pipeline 节点与 executor、聚合器，核对 PlanningEvaluator 全类。入口为 `robobench --config ... inspect-data / inference / evaluate / pipeline`，发布物为32个manifest、6092题；`pipeline --dry-run` 仅列节点。Q1使用首帧、DAG与独立judge；解析覆盖与多轮聚合须单独审计。树 `08314edc65800086656a0ff93332b023180eaa35`，代码MIT；未运行模型API或机器人。详见[完整核对记录](../papers/robo_bench_arxiv_2510_17801_full_reading_2026_11_15.md)。
