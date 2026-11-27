# PEACE: A Planner-Executor Agent with Constraint Enforcement for UAVs

- **作者：** Erdem Uysal, Timo Kehrer, Sebastiano Panichella
- **机构：** Institute of Computer Science, University of Bern（瑞士伯尔尼大学计算机科学研究所）；AI4I - The Italian Institute of Artificial Intelligence（意大利人工智能研究所，Sebastiano Panichella 兼任）
- **版本 / 载体：** arXiv:2606.00104v1 [cs.RO]（2026-05-26 提交）；Accepted to ICRA 2026 Workshop on Semantics for Reliable Robot Autonomy: From Environment Understanding and Reasoning to Safe Interaction
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2606.00104) · [arXiv HTML 全文](https://arxiv.org/html/2606.00104)
- **项目页 / 代码：** 已开源 — [GitHub: erdemuysalx/PEACE](https://github.com/erdemuysalx/PEACE)（含可运行 ROS 2 / PX4 SITL 智能体代码、系统提示词与批处理评测脚本）；配套提示词数据集已公开于 [HuggingFace: erdemuysalx/uav-mission-prompts](https://huggingface.co/datasets/erdemuysalx/uav-mission-prompts)（108 条自然语言任务指令）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§I Introduction、§II Related Works II-A~II-C、§III Approach III-A~III-F 及 Fig. 1 图注、§IV Experiments IV-A~IV-C、§V Implications and Future Directions、Acknowledgment、References [1]–[25]），并核对官方 GitHub 仓库主页 README 与文件树及 [sources/repos/paper-agentic-2606-00104.md](../repos/paper-agentic-2606-00104.md)。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [PEACE: A Planner-Executor Agent with Constraint Enforcement for UAVs](../../wiki/entities/paper-agentic-2606-00104.md)
- **研究动机与设计定位（§I–§II）：** 将基础模型驱动机器人系统归纳为三类路线——VLN（OpenUAV、NavAgent、SINGER、NaVILA、CLOI-NAV、ReMEmbR）、端到端 VLA（RT-1、RT-2、PaLM-E、CognitiveDrone、RaceVLA）与高层规划/代码生成（TypeFly、FlockGPT、Co-NavGPT、UAV-CodeAgents）。针对 ReAct 式紧耦合控制循环带来的高推理延迟与幻觉指令物理风险，以及端到端 VLA 不透明、难加显式安全约束与强依赖领域微调数据的问题，PEACE 采用单次调用（single-pass）Planner–Executor 范式。
- **核心架构与四类组件（§III-A~III-F，Fig. 1）：**
  1. **Planner–Executor：** 单次 LLM 调用将自然语言指令、机器人状态与世界模型合成为结构化 `MissionPlan`（推理说明字符串 `reasoning` + 有序工具调用列表），Executor 按序通过 ROS 2 / MAVROS / MAVLink 下发至 PX4；仅当工具执行失败时触发有界重规划（bounded replanning）。
  2. **可插拔感知与 3D 接地：** 统一检测接口支持 2D 目标检测器（如 YOLO）与 VLM 两条后端；利用同步深度图与相机内参，通过针孔相机深度投影（pinhole depth projection）将 2D 检测框反投影至 ENU 坐标系 3D 位置，并附带显式 `depth-validity` 标志区分可信与不确定定位。
  3. **世界模型与状态聚合：** `WorldModelService` 维护带时间戳的有限长度场景快照历史，`RobotStateService` 聚合里程计、飞行模式与解锁状态；每次规划（及重规划）读取一致性快照并重构用户消息。
  4. **独立几何约束强制层（Constraint Enforcement）：** 在指令抵达 PX4 前独立校验两项几何约束——高度带（超出上下限时夹紧 clamp）与起飞点水平地理围栏（越界则中止该工具调用并向重规划器返回结构化失败）。
- **实验与行为发现（§IV）：** 在 Gazebo Harmonic + PX4 SITL（`gz_x500`）中基于 108 条指令测试（takeoff and landing 12、relative motion 12、altitude and attitude control 12、waypoint navigation 13、geometric path following 13、object search and localization 12、open-vocabulary navigation 13、composite multi-step missions 13、safety violation 8；目标类别受仿真资产限制为 `{person, car, bench, traffic light, fire hydrant}`）。论文明确说明该实验为定性行为刻画而非定量基准对比，归纳出三类典型失败模式：规划器格式/规则违例、感知不确定性导致的退化计划、以及安全校验按设计触发的中止。
- **局限与后续方向（§V）：** 连续不确定感知快照缺乏工具级以下恢复机制；offboard 控制与解锁（arming）状态机耦合暴露出工具层幂等性需求；开环规划面对动态目标/障碍较脆弱；作者提出将积累的任务轨迹（规划、感知快照、执行结果）蒸馏为轻量反应式 VLA 以补充动态场景闭环能力。
