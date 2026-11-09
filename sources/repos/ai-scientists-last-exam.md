# AI Scientists' Last Exam / AISLE — 官方源码仓

- **类型：** repo（实验任务及 verifiers、agent harness、wetlab 控制器）
- **仓库：** <https://github.com/MasterAI-EAM/ai-scientists-last-exam>
- **项目主页：** <https://aisle.science/>
- **论文：** <https://github.com/MasterAI-EAM/ai-scientists-last-exam/blob/main/paper/AI_Scientists_Last_Exam.pdf>
- **许可：** Apache-2.0（第三方素材遵循原许可）
- **依赖：** Harbor 0.25.0、uv、Docker（官方 README quick start）
- **核查日期：** 2026-11-09
- **一句话说明：** 可运行的科学实验智能体评测任务、服务端验证器和实验等候/恢复工具集。
- **核对范围：** README.md；AGENTS.md；CONTRIBUTING.md；scripts/run_agent.sh；scripts/check_tasks.sh；tasks/chemistry/color_001/instruction.md、task.toml、tests/test.sh、tests/score.py、solution/solve.sh；wetlab/README.md、VALIDATION.md 与 wetlab_run.py 的主入口。

## 源码结构与真实运行入口

| 路径 | 用途 | 已核实入口 |
|---|---|---|
| README.md | Harbor 任务运行与 contribute 流程 | harbor run -p tasks/chemistry/color_001 -a oracle --env docker |
| scripts/run_agent.sh | 加载本地凭据，依次运行模型试验 | harbor run -p TASK -a terminus-2 -m MODEL |
| tasks/chemistry/color_001/instruction.md | 配方预算、工具和评分说明 | python /app/color_tool.py propose --recipe ...；history |
| tasks/chemistry/color_001/tests/test.sh | 独立评分入口 | /usr/local/bin/python -I /tests/score.py |
| tasks/chemistry/color_001/tests/score.py | 根据服务端封存实验账本评分 | 向 simulator:8000/seal 请求账本，写 /logs/verifier |
| scripts/check_tasks.sh | 任务静态检查 | bash scripts/check_tasks.sh TASK_PATH |
| wetlab/wetlab_run.py | 独立湿实验控制器 | start / continue / status / retry / abort |
| wetlab/gateway.py | 接受实验票据、实验后端回报 | 外部 gateway + 有凭证的管理员接口 |

## 运行和控制流（以官方可运行的 color_001 为例）

1. 安装 Harbor 0.25.0 并运行 oracle；以 terminus-2 和用户自有模型 API 配置执行实际 agent trial。
2. agent 读指令及只读目标数据，通过 color_tool.py 提交 200 μL 配方，最多 12 次；冻结 LightGBM 模型返回 observed_lab 与 delta_e_2000。
3. 在评测末端，tests/test.sh 启动 score.py；评分器使用服务器权威账本，不能用 agent 自写的结论或文件替代实验记录。
4. 独立 wetlab 示例则以 wait_for_result 提交实验 ticket；默认 suspend 模式先归档并移除容器，后续重建恢复会话；blocking 模式仅在官方标明支持的 harness 上运行且容器不移除。详见 wetlab/README.md。
5. wetlab 的 arithmetic simulator 示例不是物理实验，真实实验要由可信后端使用 claim_id 和 execution_id 处理，以避免重复执行；网关与设备侧需额外安全边界。

## 开源与复现边界

- 仓库公开代码、可运行入口和测试；复现 benchmark 需 Docker、依赖及模型 API，部分实验素材/基线/原始轨迹不随仓库提供。
- 官方 wetlab 验证文档 2026-11-07 记载 suspend/blocking 测试，并明确没有完成 ≥24h 的连续等待实验；当前代码对外部真实实验设备的双重执行无法作统一去重保证。
- 不可将 simulated color_001 的 LightGBM 观测称为真实湿实验数据。
- 论文全文仍未通读，仓库实现与论文全部评价口径之间的一致性待核。

## 对 wiki 的映射

- [AISLE 主实体页](../../wiki/entities/paper-ai-scientists-last-exam.md)
- [论文来源](../papers/ai_scientists_last_exam_2026.md)
- [官方网站](../sites/aisle-science.md)
