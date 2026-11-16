# Uni-Robot-Agent 官网与博客归档

- **官网：** <https://uni-robot-agent.com/>
- **文档：** <https://uni-robot-agent.com/docs/>
- **作者博客：** <https://openrua.ai/uni-robot-agent>（2026-11-03，Tsinghua University & UC Berkeley）
- **论文：** <https://arxiv.org/abs/2611.01560>；[论文归档](../papers/uni_robot_agent_arxiv_2611_01560.md)
- **代码：** <https://github.com/Open-RUA/uni-robot-agent>；[源码归档](../repos/uni-robot-agent.md)
- **核对日期：** 2026-11-11
- **开源状态：** 公开 MIT 框架、安装/运行工具和多环境适配器；真机 ARX ONE 已发布集成，Franka 仍待发布；不等于公开了新训练的机器人基座模型权重或全部实验数据。

## 作者一手资料与边界

- 官网提供 Paper / Blog / Code / Docs 导航、pip/uv 安装入口及五层接口概述；博客可阅读全文，介绍共享工具、执行流水线和 RoboDojo/Real Eval 实验协议。
- [作者博客](https://openrua.ai/uni-robot-agent) 与论文 **v2** 在真机总分一致：Qwen **53.50/100**、GPT-6 Astra **80.50/100**；微调 π₀.₅ **47.75**、Dream-1 **43.50**。Agent 使用 D405 深度，对照使用 RGB，不能忽略输入差异。早期 v1 的 Qwen 49.50 已在 v2 更正。
- 博客说明多环境统一轨迹仅为 **未来** 的开放模型训练提供基础，未声称本次已有新权重或完成跨场景训练验证。
- 在线文档首页可访问，但完整交互文档依赖 JavaScript；主要工程事实已以公开仓库 README、docs/architecture.md、docs/integrations.md 和 ARX ONE 源码 README 交叉核实。
