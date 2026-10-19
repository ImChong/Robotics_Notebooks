# ORA: Open Robot-Use Agents（官方项目页）

> 来源归档（Hao AI Lab 官方方法与项目说明）

- **标题：** ORA: Open Robot-Use Agents
- **类型：** site（官方博客 / 项目说明页）
- **发布方：** ORA Team，Hao AI Lab @ UC San Diego
- **原始链接：** <https://haoailab.com/blogs/ora/>
- **代码：** <https://github.com/hao-ai-lab/ORA>（Apache-2.0）
- **发布日期：** 2026-10-15
- **入库日期：** 2026-10-19
- **关联归档：** [博客](../blogs/ora-open-robot-use-agents.md) · [仓库](../repos/ora-open-robot-use-agents.md)
- **一句话说明：** 官方说明 ORA 的机器人 ReAct harness、工具与执行边界、仿真评测及开放源码状态。

## 结构摘要

1. **方法类别：** robot-use agent / ReAct loop；模型逐次调用有界工具，观察回执后再决定。
2. **共享核心：** 同一 agent loop、工具类型与模型，通过 benchmark boundary 和 robot card 适配不同机器人及仿真环境。
3. **评测范围：** LIBERO / LIBERO-Pro（Franka Panda）、RoboTwin（双臂 ALOHA）、RoboDojo（ARX X5）。博客报告 DeepSeek-V4.1-Flash 在 RoboTwin C2R 为 69.2%，LIBERO-Pro 为 81.4%。
4. **真机证据：** 列有初步真机演示，量化结果主体仍是仿真。
5. **开源状态：** 代码公开；仓库含运行脚本、文档、实验和 tests，Apache-2.0，第三方材料保留其原许可证。

## 对 wiki 的映射

- [ORA: Open Robot-Use Agents](../../wiki/entities/ora-open-robot-use-agents.md)
