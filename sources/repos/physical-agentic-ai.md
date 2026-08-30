# Physical Agentic AI（Liuuuxy/physical-agentic-ai）

> 来源归档

- **标题：** Retrieval-Augmented Orchestration for Multi-Robot Task Execution
- **类型：** repo
- **来源：** 亚利桑那州立大学（Arizona State University）
- **链接：** <https://github.com/Liuuuxy/physical-agentic-ai>
- **论文：** <https://arxiv.org/abs/2608.22657>
- **许可：** MIT
- **入库日期：** 2026-08-30
- **一句话说明：** 多机器人编排参考实现：G1+Go2 真机栈 + 空地 SAR Gazebo 栈；mock / live / hardware 三层。
- **沉淀到 wiki：** [`wiki/entities/paper-physical-agentic-ai.md`](../../wiki/entities/paper-physical-agentic-ai.md)

---

## 仓库入口（README）

| 组件 | 说明 |
|------|------|
| 离线单测 | `cd crew_g1_go2 && CREW_SIM=1 python3 -m pytest tests/ -q` |
| SAR 单测 | `cd sar_ws/crew_sar && SAR_SIM=1 python3 -m pytest tests/ -q` |
| mock 一致性 | `SAR_SIM=1 python3 tools/compare_mock_vs_live.py` |
| G1+Go2 mock 任务 | `CREW_SIM=1 ./sim.sh "grab from table_a then go to room_b"` |
| 真机 | `./run.sh`（需 ROS 2 Humble + Unitree 工作区；会驱动真机） |
| 仅 Go2 | `./go2_only.sh`（不创建 G1 `/user_lowcmd`） |

## 开源边界（截至 2026-08-30）

- **已开源**：编排、契约、门控、评测脚本与 203 个单测可跑。
- **真机证据**：仓内有 G1 抓取 / G1→Go2 交接视频；四条件对比表是 **mock 规划层**。
- **G1+Go2 无物理仿真器**：唯一物理仿真是 SAR Gazebo；`CREW_SIM=1` 只打桩执行。
