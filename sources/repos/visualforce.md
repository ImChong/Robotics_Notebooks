# VisualForce 官方代码

- **链接：** 论文：https://arxiv.org/abs/2610.04741；项目：https://visual-force.github.io/；代码：https://github.com/visual-force/visual-force
- **论文归档：** [论文](../papers/visualforce_arxiv_2610_04741.md)；[项目页](../sites/visualforce.md)
- **开放状态：** 代码公开，MIT；数据、检查点、SAM2 权重和机器人控制器不随仓库提供（2026-11-04）。
- **运行入口：** `third_party/forcelens_dp/scripts/berry tts` 调用 `policy_server.py`；该实现运行视觉估计并调用策略采样与候选选择，berry 默认启用夹爪反馈，与论文仅用策略生成力值筛选的描述需区分。
- **知识映射：** [VisualForce](../../wiki/entities/paper-visualforce.md)
