# Hebbian Robot-Use Agents (RUA) 官方源码
- **类型：** repo / robot-use agent
- **仓库：** https://github.com/Hebbian-Robotics/rua
- **官网：** https://hebbianrobotics.com/
- **博客：** https://hebbianrobotics.com/blog/introducing-hebbian-rua/
- **许可：** MIT；第三方组件保留各自许可
- **核对范围：** README、docs/architecture.md 和 examples/python/agent_quickstart.py；核查 Python Agent 循环及 Station、Compiler、Runtime 和 Motor Controller 边界。
- **入库日期：** 2026-11-23

## 已验证接口
- 应用侧依次调用 OpenAIRobotTools.observe、模型响应 API、robot_tools.execute。
- Station API → Robot Use Compiler → Runtime → Command Admission → Motor Controller。
- 单/双 SO-101；MuJoCo 仿真入口；真实机械臂需受控网络和人工监护。

相关：[博客和官网](../sites/hebbian-rua.md) · [独立实体](../../wiki/entities/hebbian-rua.md)。

## 2026-11-28 复核

- **核对版本：** [`9516173300984a74ce903bd3e18fbf80536a6848`](https://github.com/Hebbian-Robotics/rua/tree/9516173300984a74ce903bd3e18fbf80536a6848)（2026-11-27 的官方 HEAD）。
- **核对范围：** 通读 README、`docs/architecture.md`、`docs/control-model.md`、`docs/simulation.md`；沿 `examples/python/agent_quickstart.py` 的 `main`、`run_task`、`run_openai`、`AgentRobotTools.execute` 核对控制流。不含全库审计、模型调用或仿真/真机复现。
- **通读：** 已完成
- **运行时：** [Python Agent 示例源码](https://github.com/Hebbian-Robotics/rua/blob/9516173300984a74ce903bd3e18fbf80536a6848/examples/python/agent_quickstart.py) 先持有租约，进入 Responses API 调用后以观测 revisions 约束工具执行；并非每轮都强制单独 observe，工具结果可提供更新修订号。源码按 finish、无结果和故障/未知结果等情况结束。
- **时间与状态：** [控制合同](https://github.com/Hebbian-Robotics/rua/blob/9516173300984a74ce903bd3e18fbf80536a6848/docs/control-model.md) 记录单臂轨迹的最短时长请求而非硬截止；到达调整过的目标也可报告 completed。不确认是否执行时应核对结果和当前状态，不能把请求未收到回复当作未执行。
- **仿真与模型：** [MuJoCo 文档](https://github.com/Hebbian-Robotics/rua/blob/9516173300984a74ce903bd3e18fbf80536a6848/docs/simulation.md) 区分 station、plant 与调用端；单臂位姿策略由用户提供模型和匹配的 ONNX Runtime 库。
