# Eden：Embodied-AI Development Environments Nexus
- **类型：** 开源机器人学习框架
- **仓库：** https://github.com/embodied-ai-nexus/Eden
- **文档：** https://embodied-ai-nexus.github.io/Eden/
- **发行包：** eden-world 0.1.1（PyPI）
- **许可：** Apache-2.0
- **核对范围：** 官方 README、pyproject.toml、eden/tasks/benchmark/reacher/config.py 的任务注册、奖励与管理器配置；核对文档提供的 CLI 命令及项目入口。
- **通读：** 已完成
- **入库日期：** 2026-11-23

## 已验证接口
- pip install eden-world；eden list；eden train --task reacher
- eden inference --task reacher --checkpoint ... --vis
- eden deploy --task reacher --checkpoint model.pt --mujoco
- eden export --task reacher --checkpoint model.pt
- Genesis 1.4.7；Python 3.12–3.13；类型化 MDP 任务配置。README 示例以 4096 env 训练 reacher，并非本次独立性能实测。

相关：[官网](../sites/eden-embodied-ai-nexus.md) · [实体](../../wiki/entities/eden-embodied-ai-nexus.md)。
