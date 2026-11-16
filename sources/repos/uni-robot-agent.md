# uni-robot-agent 源码归档

- **仓库：** <https://github.com/Open-RUA/uni-robot-agent>
- **论文：** <https://arxiv.org/abs/2611.01560>；[论文归档](../papers/uni_robot_agent_arxiv_2611_01560.md)
- **项目页：** <https://uni-robot-agent.com/>；[官网与博客归档](../sites/uni-robot-agent.md)
- **版本：** 官方 main commit `d5e8085cebc05e59aaf9e237e24c7a59b1a59708`（2026-11-11）；README 标注 **v0.2.0**（2026-11-11），快照并非永久固定 API。
- **许可：** MIT；外部驱动/资产/模型遵循其各自许可。
- **核对日期：** 2026-11-11

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
- 本次只核对静态仓库与论文，**没有实际部署或复现实验**。
