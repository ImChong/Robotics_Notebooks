# uni-robot-agent 源码归档

- **仓库：** <https://github.com/Open-RUA/uni-robot-agent>
- **论文：** <https://arxiv.org/abs/2611.01560>；[论文归档](../papers/uni_robot_agent_arxiv_2611_01560.md)
- **项目页：** <https://uni-robot-agent.com/>；[官网与博客归档](../sites/uni-robot-agent.md)
- **版本：** 本次核对官方 main commit `b269a52156d22782a5e21833f29d15e4edc26506`（2026-11-17）；最新公开 Release **v0.2.0**（2026-11-11）。Piper-X 只在 main 源码，**不随 v0.2.0 包发布**；两者不得混为一谈。
- **许可：** MIT；外部驱动/资产/模型遵循其各自许可。
- **核对日期：** 2026-11-21

## 已读源码及真实调用链

1. `uni_robot_agent/cli.py` 解析 `run` 命令，进入 `session.run(args)`；`Session.run_agent()` 负责执行与收尾。
2. `session.assemble_session(args)` 加载选定 integration，生成 session，注册共享工具（`Toolkit.register_tools`），由 `create_executor` 创建对应执行器并连接 Agent runtime。
3. `execution/parallel.py` 的 `ParallelToolExecutor.call/submit` 经 `ResourceScheduler` 按资源策略调度，后台调用返回 job handle，`job_result` 取回结果。
4. `tools/toolkit.py` 的 `execute_tool` 根据注册的 `ToolDef` 调用 `tool_def.handler(**input_dict)`，构造或校验 `ToolOutput` 并记录结果。`execution/scheduler.py` 含 exclusive resource worker 与排队取消逻辑。
5. `docs/architecture.md` 明确 Agent / Orchestration / Tool / Service / Backend 五层；以上为源码证实的控制主链，具体仿真/真机 handler、通信与模型实现由 integration 决定。

## 使用与开源范围

- 官方 README：Python 3.11 Linux GPU 主机可 `pip install uni-robot-agent`，然后 `uni-robot-agent setup --integration robodojo`；运行需自备 provider endpoint/API Key 与可用的目标环境。
- RoboDojo 默认启用 depth。要接近论文无特权深度主实验，需传 `--no-depth` 并保留 camera calibration；其他参数见 [集成文档](https://github.com/Open-RUA/uni-robot-agent/blob/main/docs/integrations.md)。
- README 还列出 RoboUse、RoboLab、RoboTwin 2.0、LIBERO、DexJoCo、IsaacLab Arena、NaVILA-Bench 等集成；Hybrid 支持 G1/Go2 等示例，具体依赖与成熟度不同，不应宣称已通过所有论文实验。
- **真机发布边界已变化：** [ARX ONE](https://github.com/Open-RUA/uni-robot-agent/blob/main/uni_robot_agent/integrations/arx_one/README.md) 现公开双 ARX X5 + 三台 RealSense D405 的 device-process、environment 与部署流程；Franka 仍在 README 标为 coming soon。ARX 文档明确启动控制器可能触发归零或闭合夹爪，部署必须遵循操作者的硬件安全与校准流程。
- **Piper-X 补充（main，非 v0.2.0 发布包）：** [集成说明](https://github.com/Open-RUA/uni-robot-agent/blob/main/uni_robot_agent/integrations/piper_x/README.md)、[2026-11-17 上游主干](https://github.com/Open-RUA/uni-robot-agent/commit/b269a52156d22782a5e21833f29d15e4edc26506)；支持 SocketCAN 双臂与 RealSense RGB-D、共享 Direct 工具及 `readonly` 观测模式，需源码安装、实体标定与独立 Body 服务。官方 `CHANGELOG.md` 将其列在 **Unreleased**，而不是 v0.2.0。
- **Piper-X 源码入口：** `integrations/piper_x/__main__.py` 负责离线 `validate`/`host-config` 与环境启动，`environment/server.py` 创建 `EnvironmentService` 并处理取消，`environment/adapter.py` 实现 `step_path`，`environment/control.py` 用 Body RPC/租约及操作回执处理运动与停止；`session.py` 拒绝在观察-only 环境暴露运动工具。
- **真实运行限制：** `command_limits` 不做桌面/自碰/双臂互碰校验；`execution_uncertain` 不能自动重放，必须保留人工恢复与物理急停。官方 README 报告两次 2026-11-13 受监督真机试验（一通过，一次右臂末段跟踪 **30.66 mm > 30 mm** 阈值而拒绝），属于有限站点观察，非可靠性统计，也非论文 19 任务结果。
- **安装兼容：** v0.2.0 的主包与原生 runtime 需要版本匹配、分开安装并同步重启；发布说明列出 environment protocol 4、tool schema 8、actor HTTP v3，旧服务不兼容。
- 本次只核对静态仓库与论文归档，**没有实际部署或复现实验**。
