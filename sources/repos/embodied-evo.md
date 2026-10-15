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
