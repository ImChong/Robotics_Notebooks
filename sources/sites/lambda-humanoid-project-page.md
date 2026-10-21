# λ₀ 官方项目页核查

> 来源归档（site / Lambda Humanoid）

- **URL：** <https://lambdahumanoid.github.io/>
- **标题：** λ₀: Scaling Open-World Egocentric Human Data for Humanoid Loco-Manipulation
- **核查日期：** 2026-10-16
- **论文：** <https://arxiv.org/abs/2610.18950>
- **代码：** <https://github.com/LambdaHumanoid/openlambda>
- **模型权重：** <https://huggingface.co/lambda-humanoid/lambda0-midtrain-100k>
- **样例数据：** <https://huggingface.co/datasets/lambda-humanoid/HumanVerse-500-sample>（核查时 Hugging Face 返回 401 Unauthorized）
- **开放状态：** 代码仓库及 Stage II checkpoint 已公开；完整 HumanVerse 数据未在代码仓库发布，项目页链接的样例数据目前需授权访问；Stage III 后训练流程仍在准备中。
- **源码归档：** [OpenLambda 仓库核查](../repos/openlambda.md)
- **论文归档：** [arXiv:2610.18950](../papers/lambda0_arxiv_2610_18950.md)
- **Wiki 主实体：** [λ₀ / OpenLambda](../../wiki/entities/paper-lambda0-egocentric-human-pretraining.md)

## 项目页内容

项目页展示 HumanVerse 人类数据采集、从人体动作到 G1 的动作对齐流程、Stage I/II/III 训练结构及 G1 真机演示。页面列出的真实任务结果为 62.5% 完成率和 80.4% 平均任务进度；人体采集是 robot-free，但 Stage III 使用人类任务示范与原生机器人示范进行机器人适配。

## 开源核查

官网直接链接 OpenLambda 代码、arXiv 论文和 Hugging Face 权重/样例数据。仓库说明可见人体采集、动作对齐、仿真重放、硬件环境和 λ₀ 模型训练包；Hugging Face 模型卡将权重标为 Apache-2.0，并说明是未含 EMA 的 Stage II 100k checkpoint，不是部署策略。README 明确完整 Stage III 配方尚在准备、预训练和中训练数据不包含在代码发布中。故记为**代码与中训练 checkpoint 已发布、数据部分/未开放、真机后训练流程未完整发布**。
