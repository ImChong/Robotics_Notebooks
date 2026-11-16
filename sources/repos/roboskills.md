# RoboSkills：OpenRoboKit 官方 Agent 工作流与工具归档

- **类型：** repo / agent workflows / embodied robotics
- **官方仓库：** https://github.com/openrobokit/roboskills
- **项目入口：** https://openrobokit.com/roboskills
- **工作流总览：** https://github.com/openrobokit/roboskills/blob/main/README.md
- **Agent as Policy 文档：** https://github.com/openrobokit/roboskills/blob/main/src/roboskills/workflows/agent_as_policy/README.md
- **Agent as Policy 可运行入口：** https://github.com/openrobokit/roboskills/blob/main/src/roboskills/workflows/agent_as_policy/run.py
- **Agent as Policy 指令：** https://github.com/openrobokit/roboskills/blob/main/src/roboskills/workflows/agent_as_policy/SKILL.md
- **Real2Sim 文档：** https://github.com/openrobokit/roboskills/blob/main/src/roboskills/workflows/real2sim/README.md
- **许可证：** MIT（仓库 README 与 pyproject.toml 标注；第三方模型各守自己的许可）
- **核查日期：** 2026-11-16
- **核对范围：** 完整阅读官方 README、pyproject.toml、agent_as_policy/README.md、agent_as_policy/SKILL.md、agent_as_policy/run.py 以及 real2sim/README.md；并核对 OpenRoboKit 官方组织介绍。未逐文件审计其他四类工作流、第三方依赖和所有 benchmark 适配器。
- **通读：** 已完成

## 官方主张与可核实入口

- 五个工作流：**Auto Research**（反复仿真改进）、**Agent as Policy**（LLM 每轮动作决策）、**Agentic Simulation**（自然语言仿真任务）、**Real2Sim**（视频/示范重建为物理仿真）、**Motion Generation**（文本/图像/视频到运动视频）。
- 安装条件：Python >=3.10；`uv add roboskills` 或 `pip install roboskills`。完整工作流要求克隆仓库、`uv sync`，部分工具还需模型权重、GPU、外部服务或机器人驱动。
- `agent_as_policy/run.py` 是真实 Python CLI，`main()` 解析 `--adapter`、`--out`、`--model` 与 adapter 参数；`local()` 加载 adapter，`make_envs()` 初始化向量环境，`ThreadPoolExecutor` 为每个 episode 调用 `agent.run`。
- agent 内部通过 `Repl` 向 `bot` 工具发送观察、分割、规划与执行等调用；`execute` 才使机器人动作。共享 planner lock 仅保护 GPU 规划与代码调用，环境动作/观察期间可释放；结束写各 episode `result.json` 和总体 `summary.json`。
- Real2Sim 文档说明重建模块包含 SAM 3、MoGe-3、GeoCalib、SAM 3D、HaWoR、对齐、跟踪、CoACD、MuJoCo 等可选阶段，物理侧可通过 RoboKit IK / SimMPC 追踪失败的接触片段。不同 `--tools` 参数代表不同实际流水线，不应说所有阶段必跑。

## 论文/性能与边界

- RoboSkills README 的成本数字是作者指定工作流与模型的案例，不是所有任务都能保持同等费用/成功率；本次未调用外部 LLM、没有运行仿真或真机。
- RoboSkills 是 OpenRoboKit 官方组织列出的**第四个独立项目**，并非 RoboKit 论文中新增的优化器；不能把其 Agent benchmark 计入 CoRL RoboKit 论文实验。
- 官方 PyPI 安装只提供基础包，外部仿真、渲染、感知和真机 SDK 需逐项确认依赖与许可证。

## 对 wiki 的映射

- [RoboSkills 详情](../../wiki/entities/roboskills.md)
- [OpenRoboKit 全景](../../wiki/overview/openrobokit-toolkit.md)
