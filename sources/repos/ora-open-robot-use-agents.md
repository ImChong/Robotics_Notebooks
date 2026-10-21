# ORA: Open Robot-Use Agents

> 来源归档（官方代码仓库）

- **标题：** ORA: Open Robot-Use Agents
- **类型：** repo
- **链接：** <https://github.com/hao-ai-lab/ORA>
- **许可证：** Apache-2.0（README 提示第三方代码保留原许可）
- **入库日期：** 2026-10-19
- **关联资料：** [官方项目页](../sites/ora-open-robot-use-agents.md) · [官方博客](../blogs/ora-open-robot-use-agents.md)
- **一句话说明：** 开源机器人使用 agent harness，将模型 API、机器人任务上下文、感知/动作工具、动作监督器和 benchmark runtime 接成可回放的逐步闭环。

## 仓库入口

- `ora/`：agent 与运行时实现
- `examples/`、`docs/`：示例与使用文档
- `experiments/`：评测和实验
- `scripts/ora`：环境安装与任务启动入口
- `ora-tui/`：终端交互界面
- `tests/`：测试

## 复现与部署边界

README 提供锁定环境安装与运行流程（`uv sync --all-extras --locked`）。单 GPU 环境可配置 LIBERO、RoboTwin、RoboDojo；官方验证硬件包括 RTX PRO 6000 Blackwell（端到端）与 H200（launch checks）。运行需模型服务 API key；无 GPU 本地路径只覆盖明确支持的任务，不能据此推断所有 benchmark 可在 CPU 复现。README 称 PyPI 发布待完成，当前应从源码 checkout 启动。

## 对 wiki 的映射

- [ORA: Open Robot-Use Agents](../../wiki/entities/ora-open-robot-use-agents.md)
