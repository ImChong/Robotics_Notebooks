# ASENA 官方项目页

> 来源归档（site；2026-10-16）

- **项目：** ASENA: Self-evolving Agents for Embodied Navigation
- **官方项目页：** <https://research.nvidia.com/labs/gear/asena/>
- **论文：** <https://arxiv.org/abs/2609.39207>
- **HTML（v2）：** <https://arxiv.org/html/2609.39207v2>
- **机构：** NVIDIA GEAR Lab；论文作者还列有 UC San Diego 与 UT Austin
- **官方代码 / 权重 / 数据：** 本次核查的项目页包含系统方法、演示、数据集说明与评测结果，未列出 GitHub、权重或数据下载链接；没有确认到官方可运行代码仓库。
- **独立详情：** [ASENA](../../wiki/entities/paper-asena-self-evolving-agents.md)

## 页面核查

- 项目页展示 robot harness、可选 ASENA-VLN、原子导航数据、基准结果、工作区演进与 Unitree G1 演示。
- 论文 v2 将物理执行描述为受监督执行：生成动作经过在线检查并需操作者批准；另有导航安全限制、心跳停止与手动急停。
- 论文补充材料包含 Jetson AGX Thor 上较早 ASENA-VLN checkpoint 的量化推理测量；这不是代码或模型权重发布。
- 截至 2026-10-16，项目页没有给出官方训练 / 推理代码、公开 checkpoint 或数据集下载入口。