# Hebbian Robot-Use Agents (RUA) 官方源码
- **类型：** repo / robot-use agent
- **仓库：** https://github.com/Hebbian-Robotics/rua
- **官网：** https://hebbianrobotics.com/
- **博客：** https://hebbianrobotics.com/blog/introducing-hebbian-rua/
- **许可：** MIT；第三方组件保留各自许可
- **核对范围：** README、docs/architecture.md 和 examples/python/agent_quickstart.py；核查 Python Agent 循环及 Station、Compiler、Runtime 和 Motor Controller 边界。
- **通读：** 已完成
- **入库日期：** 2026-11-23

## 已验证接口
- 应用侧依次调用 OpenAIRobotTools.observe、模型响应 API、robot_tools.execute。
- Station API → Robot Use Compiler → Runtime → Command Admission → Motor Controller。
- 单/双 SO-101；MuJoCo 仿真入口；真实机械臂需受控网络和人工监护。

相关：[博客和官网](../sites/hebbian-rua.md) · [独立实体](../../wiki/entities/hebbian-rua.md)。
