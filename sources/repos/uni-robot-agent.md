# uni-robot-agent 源码归档

- **仓库：** <https://github.com/Open-RUA/uni-robot-agent>
- **论文：** <https://arxiv.org/abs/2611.01560>；[论文归档](../papers/uni_robot_agent_arxiv_2611_01560.md)
- **项目页：** <https://uni-robot-agent.com>；[项目归档](../sites/uni-robot-agent.md)
- **版本：** commit `9d6a051e2045fa23ad863cfed49101e699d520de`，核对日期 2026-11-03。
- **许可：** MIT，外部组件各自许可见仓库。

## 已核对入口

README 给出 `uni-robot-agent run --integration robodojo ...`。`uni_robot_agent/cli.py` 将 run 命令交给 `session.run`；`session.py` 组装 toolkit 与 `ParallelToolExecutor`，调用 agent；`execution/parallel.py` 调度工具；`tools/toolkit.py` 调用注册的 handler。运行依赖模型服务和环境，未在本次运行仿真。

## 开放边界

README 列出 RoboDojo、RoboUse、RoboLab 仿真集成；ARX One 与 Franka 真机连接标注 coming soon，公开包尚无受支持的真机连接流程。仓库及其 README 可访问，项目站点访问失败，站点列出的其他资源尚待核查。
