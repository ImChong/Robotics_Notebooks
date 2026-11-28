# robodeltas/inspect-robots：CoRL-harness-rework

> 一手仓库归档

- **类型：** repo
- **代码：** <https://github.com/robodeltas/inspect-robots>（默认分支 CoRL-harness-rework）
- **快照：** [f00acc1，2026-10-13](https://github.com/robodeltas/inspect-robots/tree/f00acc1876125dd1ba072e7159ae6944e15ce7a4)
- **文档：** <https://docs.inspectrobots.org/>；[本分支内文档](https://github.com/robodeltas/inspect-robots/tree/f00acc1876125dd1ba072e7159ae6944e15ce7a4/docs/guide)。托管站在本次环境未能成功读取，以提交内文档为准。
- **License：** MIT；框架 Alpha；核心运行依赖 NumPy
- **核对范围：** 默认分支 README、pyproject/许可、核心 guide（概念、策略/本体、快速开始、评分、日志与 Rerun），CLI、eval、rollout、JSON sink 和 Agent setpoint 路径；以 GitHub 元数据及提交比较核对两仓关系；未运行第三方模型或真机。
- **通读：** 已完成
- **沉淀到 wiki：** [Inspect Robots](../../wiki/entities/inspect-robots.md)

## 仓库关系

GitHub API 将 [robodeltas 分支仓](https://api.github.com/repos/robodeltas/inspect-robots) 和 [Robocurve 主仓](https://api.github.com/repos/robocurve/inspect-robots) 登记为不同的仓库 ID，前者也未标为 GitHub fork；不应写成同一 URL 的重定向、重命名或用分支快照代表主仓最新版。[提交比较](https://api.github.com/repos/robodeltas/inspect-robots/compare/d08442a9d1f43af4658c8d71e02d461e780286e1...f00acc1876125dd1ba072e7159ae6944e15ce7a4) 显示 f00acc1 比既有归档的 d08442a 快照多 9 个提交，后者是合并基点，说明核心实现有明确历史继承关系；因此两份资料归到同一个 Inspect Robots 详情节点。旧版本与 G1 适配历史见 [Robocurve 主仓归档](./robocurve_inspect_robots.md)。

## 核心源码与证据

| 核对位置 | 能验证什么 |
|---|---|
| [CLI 与 eval](https://github.com/robodeltas/inspect-robots/blob/f00acc1876125dd1ba072e7159ae6944e15ce7a4/src/inspect_robots/cli.py) · [评测循环](https://github.com/robodeltas/inspect-robots/blob/f00acc1876125dd1ba072e7159ae6944e15ce7a4/src/inspect_robots/eval.py) | 解析已注册的任务、策略、本体；空间检查成功后才进入 rollout；以分场景/重复试验的记录评分与汇总 |
| [执行循环](https://github.com/robodeltas/inspect-robots/blob/f00acc1876125dd1ba072e7159ae6944e15ce7a4/src/inspect_robots/rollout.py) · [JSON sink](https://github.com/robodeltas/inspect-robots/blob/f00acc1876125dd1ba072e7159ae6944e15ce7a4/src/inspect_robots/logging/json_log.py) | 策略先经 Controller 取动作、Approver 审查，才 step；rollout 自身不按墙钟限速，控制节拍由本体适配层等实现；JSON 与可选 Rerun 另有职责 |
| [分支 HEAD 改动](https://github.com/robodeltas/inspect-robots/commit/f00acc1876125dd1ba072e7159ae6944e15ce7a4) · [Agent 工具](https://github.com/robodeltas/inspect-robots/blob/f00acc1876125dd1ba072e7159ae6944e15ce7a4/plugins/inspect-robots-agent/src/inspect_robots_agent/_tools.py) | 绝对动作模式在有 setpoint 时可显示持有的命令目标及可比较的 readback 差值；位移模式无 setpoint，形状或数值不合法时不做减法；不是接触或成功判定 |
| [Quickstart](https://github.com/robodeltas/inspect-robots/blob/f00acc1876125dd1ba072e7159ae6944e15ce7a4/docs/guide/quickstart.md) · [评分指南](https://github.com/robodeltas/inspect-robots/blob/f00acc1876125dd1ba072e7159ae6944e15ce7a4/docs/guide/scoring.md) | 内置 CubePick mock 可从纯 Python 走完接口；异常回合保留状态但不计入分数，弃权不纳入平均，不能只报幸存分母 |

公开的是评测核心与若干插件；外部策略权重、服务、具体本体适配及基准任务需要按各自入口与许可另行核对。
