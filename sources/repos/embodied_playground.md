# Embodied Playground | Official environment release

- **类型：** repo
- **代码：** [固定提交 45993d4](https://github.com/Winniechen2002/Embodied-Playground/tree/45993d432ae58a9f3dd05dc2ff3c6e16eb39198f)
- **项目页：** <https://winniechen2002.github.io/Agent-bench-website/>
- **论文：** <https://arxiv.org/abs/2610.27039v2>
- **许可：** 原创代码 Apache-2.0，原创非代码资产与文档 CC BY 4.0；继承与第三方资产保留其各自许可。见固定提交的 `LICENSING.md`。
- **核对范围：** 官方 README、Agent 指南、环境启动、SDK、协议实现与评测清单；未启动大型仿真或调用模型。
- **通读：** 已完成
- **最初入库 / 本次核对：** 2026-10-31 / 2026-11-25
- **对应节点：** [Embodied Playground](../../wiki/entities/paper-embodied-playground.md)

| 入口 | 职责 |
|---|---|
| `scripts/environment_runtime.py` | 根据任务标识和正式 trial 选择后端、权限和预算，监督单次回合墙钟时限；部分 Isaac 任务分派专用运行入口 |
| `source/agentbench_sdk/agentbench_sdk/client.py` | 在隔离会话中发送 describe、observe、step 与 finish |
| `source/agentbench_platform/agentbench_platform/protocol.py` | 公开接口、权限过滤及回合收尾 |
| `scripts/environment_client.py` | 连接与中性动作检查，不代替学习或 agent 评测 |
| `evaluation_protocol.py` 与 `Agent.md` | 注册条件、完整覆盖与报告约定；宿主负责启动被测会话，不是仓库内置模型服务 |

本地无模拟器清单检查通过：410 个任务，每组四个注册条件；含两种发射任务起始方式的完整程序清单为 510 组 / 2,040 槽，不是公开 loaded-only Full test 的 1,640 条件，也不是 EP-100 的 400 条件。

[论文归档](../papers/embodied_playground_arxiv_2610_27039.md) · [项目页归档](../sites/embodied-playground.md)
