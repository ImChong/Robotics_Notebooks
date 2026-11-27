# RobotWorld: Benchmarking Multimodal Agents for Robot Use Across Diverse Tasks and Embodiments

- **作者：** Zhiqin Yang, Chenxin Li, Xiaomeng Hu, Yibin Liu, Weidong Huang, Jiankai Sun, Haitao Li, Zijian Wu, Yuzhi Huang, Fanding Huang, Hanwen Sun, Jiashun Liu, Jingqi Tong, Mingxin Huang, Shaoli Hu, Shijue Huang, Tianyi Bai, Xinyuan Wang, Yunlong Lin, Zhengyang Tang, Zhexin Zhang, Zhuo Chen, Xierui Song, Juntao Dai, Boyuan Chen, Jiaming Ji, Fangneng Zhan, Mengkang Hu, Wei Xue, Yonggang Zhang, Han Hu, Tsung-Yi Ho, Yike Guo
- **机构：** arXiv HTML 原文未列明作者所属机构（仅附通讯邮箱 `robotworld.ai@gmail.com`）
- **版本 / 载体：** arXiv:2610.10409v1 [cs.RO]（2026-10-07 提交，62 页，25 图）
- **原文：** [arXiv HTML v1](https://arxiv.org/html/2610.10409) / [arXiv Abs](https://arxiv.org/abs/2610.10409)
- **项目页 / 代码：** 未确认开源代码（arXiv:2610.10409v1 原文及抓取元数据未公开独立项目页或官方 GitHub 仓库链接；§5.6 与 Appendix D.3 提及项目网站统计口径，Appendix B 提及归档快照 `snapshot-20261006T025143Z` 与补充接口清单 `appendix_interfaces.json`，但未提供公开代码库入口）
- **核对范围：** arXiv:2610.10409v1 全文（Abstract、§1–§8、Table 1–5 / 11–14 / 16、Fig. 1–25 文字说明及 Appendix A–H 全部任务画像、工具注册目录、提示词摘录、全量 84 任务结果表、成功轨迹与失败/裁定记录）
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **核心问题与基准定位（§1–§4）：** 通用多模态智能体已能在数字环境中写代码与调用工具，但尚不清楚这些能力在多大程度上迁移到物理世界控制。作者提出 **RobotWorld**，汇总 **20 个源项目共 84 项任务**，覆盖 **操作（38）、移动操作（20）、足式与全身运动（11）、驾驶（11）、空中飞行（4）** 五大领域及七类本体。
- **统一交互契约与工作区边界（§3, Appendix B–C）：** 每个任务形式化为 $\mathcal{T}_i = (\mathcal{E}_i, \rho_i, g_i, O_i, A_i, C_i, B_i, V_i)$，在前沿执行框架（frontier harness）中交替运行「只读 `/observations` + 离线 `/workspace`（shell 与图像查看，期间物理仿真暂停）」与「有界机器人工具调用（推进物理并计入控制步 $n_{k+1} = n_k + d_k$）」。5 模型统一评测中关闭环境侧 `coding_control`（每步回调生成程序控制），仅允许直接动作工具调用（如 `move_eef`、`move_robot`、`move_base`、`drive`、`apply_action` 等）；历史窗口保留当前观测及每两轮采样的至多 4 条历史观测。
- **跨适配器接口差异与成功判据（§4.3, Appendix B.2–B.7）：** 20 个源组虽然共享框架外衣，但同名工具语义各异（例如 RoboDojo `move_eef` 控制世界系夹爪夹持点且 `0` 为闭合，RoboLab `move_eef` 控制根系法兰位姿且 `gripper_close=0` 为张开，RoboCasa `move_eef` 为基座系归一化 OSC 增量）；13 个原生向量组暴露 `observe` + `apply_action`（4–52 维）。68 项任务采用原生成绩判定（`N`），16 项采用 `world-state-v1`（`W`），覆盖位置姿态、物体状态、接触释放、持续保持时间窗（如 2 s 联合稳定）、事件顺序与路径安全六类证据。
- **五模型评测协议与主结果（§5, Table 3, Table 11–14, Appendix G–H）：** 2026-09-30 至 2026-10-06 评测 5 个模型各 1 次/任务共 420 条轨迹；非动作交互预算为连续 15 次、总计 30/60/120 次（排球 1v1 为 20/150）。
  - 总体成功率（Table 3）：**GPT-6 Astra 16/84（19.0%）**，**Claude Opus 5.5 13/84（15.5%）**，**Kimi K3 2/84（2.4%）**，**DeepSeek V4.1 Flash 1/84（1.2%）**，**Gemini 3.8 Flash 1/84（1.2%）**；五模型并集仅 **21/84（25.0%）**，其余 **63 项任务全部失败**。
  - 领域分化：Astra 擅长桌面操作（9/38）与移动操作（4/20）；Opus 5.5 在移动操作为 0/20，但空中任务达 **3/4（75.0%）** 且取得唯一足式任务成功（`quadruped_push_recovery` 1/11）。
  - 探索性能力需求分组（§6.6, Fig. 19）：Astra 在空间目标（13/45 vs 6/45）与受限接触（6/41 vs 1/41）领先；Opus 5.5 在连续平衡/跟踪（4/20 vs 1/20）与定时交互（4/7 vs 2/7）领先。
  - 资源消耗与回溯裁定（Table 12–14, Appendix H.5）：Astra 估算费用 $9,912.80（941M tokens，54.6 h）；Opus 5.5 $2,916.43（1.63B tokens，140.3 h）；Kimi K3 $624.09（外推 2.00B tokens，163.8 h）。共有 21 条记录经过预算边界回溯裁定（其中 Opus 5.5 有 3 条在超限后才完成任务的记录被改判为失败）。
- **核心失败机制（§6.2, §6.4）：** (1) 位姿到达但丢失物体状态（如 Kimi 倒水第 42 步碰倒水瓶至 400 步未能复抓）；(2) 准备与分析阶段耗尽预算（如 Kimi 切菜 2,000 步全耗在进厨房，DeepSeek 在 Bench2Dex task-232 零动作步耗尽 15 次连续非动作预算）；(3) 误判完成而停止修正（如 Kimi 扫方块第 861 步自称完成，DeepSeek 导航自称无需移动后连续发 105 步零速指令）；(4) 恢复过晚或不满足持续时间窗（如倒立摆末端误差 2.45 cm 却只稳住 0.565 s < 1 s）。
- **对 wiki 的映射：** [RobotWorld: Benchmarking Multimodal Agents for Robot Use Across Diverse Tasks and Embodiments](../../wiki/entities/paper-agentic-2610-10409.md)
