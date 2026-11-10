# Einsia/EmbodiedEvo（官方源码）

> 来源归档（repo）

- **代码：** <https://github.com/Einsia/EmbodiedEvo>
- **项目页：** <https://lab.einsia.ai/embodiedevo/>
- **论文：** <https://arxiv.org/abs/2610.18867>
- **类型：** 具身 coding-agent harness / RoboDojo-Sim 仿真
- **核查日期：** 2026-10-15

## 一句话摘要
提供将 agent 编写的 Python 控制程序接入 RoboDojo 的运行框架，并支持在 Playground 积累 Skills / Lessons、冻结后由新 agent 在 Test 场景复用。

## 公开代码与目录
官方 README 将主要模块概括为：
| 路径 | 作用 |
|------|------|
| orchestrator/ | 任务配置、agent 隔离、会话、冻结和 Test 生命周期 |
| environments/、simulators/ | Python 执行、原生交互原语和 RoboDojo 适配 |
| integrations/robodojo_policy/ | XPolicyLab policy 与原生评测传输 |
| tasks/RoboDojo/ | Playground / Test 模板、原语手册与公开 recipes |
| docker/agent/、scripts/ | thin agent 镜像、环境准备和诊断 |
| tests/ | CPU 测试；部分 transport 测试需要部署好的 XPolicyLab |

README 提供安装 host requirements 与开发包、构建 docker/agent 镜像，以及运行 embodiedevo explore / test 的说明。执行和评测依赖单独准备的模拟器环境，源码仓本身不能证明论文结果开箱可复现。

## 已核对的源码入口与真实调用关系（2026-11-10）
- [`pyproject.toml`](https://github.com/Einsia/EmbodiedEvo/blob/main/pyproject.toml) 声明 `embodiedevo = orchestrator.__main__:main`；`explore`/`test`/`run` 进入 `orchestrator.session_runner`。
- [`session_runner.py`](https://github.com/Einsia/EmbodiedEvo/blob/main/orchestrator/session_runner.py) 的 `explore` 调用 `run_episode`，从 Playground 工作区仅拷贝 `skills/`、`lessons/` 到 `frozen_harness/` 并记录 hashes；`test` 为每个 seed 创建独立 scene/agent，传入冻结目录。
- [`tasks/RoboDojo/Workspace/scripts/env.py`](https://github.com/Einsia/EmbodiedEvo/blob/main/tasks/RoboDojo/Workspace/scripts/env.py) 将 `submission/solution.py` 与 `--include` 所选 skills 的 Python 文本经 Unix socket 送至宿主 [`orchestrator/session.py`](https://github.com/Einsia/EmbodiedEvo/blob/main/orchestrator/session.py) 的 `Session.handle`。
- `Session.handle` 调用 `environments.execute`；[`simulators/execution.py`](https://github.com/Einsia/EmbodiedEvo/blob/main/simulators/execution.py) 的 `CodeTask._step` 执行代码，L1 通过 [`RoboDojoLowLevelApi`](https://github.com/Einsia/EmbodiedEvo/blob/main/simulators/robodojo_api.py) 的 `step(action)` 下发至 native `control_step`。会话保留 Python 全局变量与机器人状态，不因修改下一段程序而回滚动作。
- `Session.handle` 明确拒绝 Test 阶段的 `reset`；`create_workspace` 为 Test 拷贝冻结知识，协议在每次 Test 前后校验 harness hash。**此处是运行时源码路径核验，并非复现论文分数。**

## 开源与许可边界
- GitHub 上可访问源码 checkout，README 包含设置和运行说明。
- **项目级许可证尚未选择**（按 2026-10-15 README 核查）；不要默认代码可按任意开源许可证再分发。
- 运行依赖 Linux、Python 3.11+、Docker、RoboDojo / Isaac Sim 环境与资产、NVIDIA GPU 和模型凭据；模拟器与资产不随源码分发。
- 论文评测是 RoboDojo-Sim 仿真，不是已证实的真机部署。

## 对 wiki 的映射
- 实体页：[EmbodiedEvo](../../wiki/entities/paper-embodied-evo-coding-agent.md)
- 项目页：[归档](../sites/embodied-evo.md)
- 论文：[归档](../papers/embodied_evo_arxiv_2610_18867.md)

## 原始来源
- 官方仓库 README：<https://github.com/Einsia/EmbodiedEvo>
