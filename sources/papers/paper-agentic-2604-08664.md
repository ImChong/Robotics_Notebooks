# GenPHRI: Agentic Generative Simulation for Physical Human-Robot Interaction

- **作者：** Junxiang Wang, Nir Pechuk, Xinwen Xu, Tiancheng Wu, Nayon Kim, Julian Millan, Zackory Erickson
- **机构：** Carnegie Mellon University (Robotics Institute)
- **版本 / 载体：** arXiv:2604.08664v2 [cs.RO]（2026-04-09 提交 v1，2026-09-22 修订 v2）；14 页，7 图，7 表
- **原文：** [arXiv HTML (2604.08664)](https://arxiv.org/html/2604.08664) ；[arXiv Abstract](https://arxiv.org/abs/2604.08664)
- **项目页 / 代码：** [GenPHRI 项目页](https://rchi-lab.github.io/gen_phri/) ；宣称将开源或未列链接（论文摘要称发布 GenPHRI 平台，但截至 2026-11-27 论文全文与项目页仅展示 50 个交互式场景、生成修订回放及真机演示视频，未提供独立源码仓库地址；原文引用的 `https://github.com/Genesis-Embodied-AI/Genesis` 为底层物理仿真引擎依赖，非 GenPHRI 本体代码）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§1–§6、Limitations and Future Work、References、Appendix A–F，含 Table 1–7 与 Figure 1–7 图注）及项目页公开文本；已核查 `candidate_repos` 中的 `Genesis-Embodied-AI/genesis-world` 确认其为通用仿真平台依赖而非本项目官方代码仓库。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [GenPHRI 实体页](../../wiki/entities/paper-agentic-2604-08664.md)
- **问题定义与场景表示：** 给定固定任务规范 $\tau$（含任务类型、人体姿态、支撑家具与目标肢体段），寻找场景配置 $x = (A_{\mathrm{room}}, A_{\mathrm{pose}}, A_{\mathrm{position}}, A_{\mathrm{robot}}, A_{\mathrm{motion}})$，在满足跨阶段硬约束 $c_j(x;\tau)=1$（如所有路点可达、底座倾角 $\le 1^\circ$、接触速度 $< 0.05\text{ m/s}$ 等）的前提下最大化任务质量效用 $U_\tau(x)$。
- **技术栈与模型配置：**
  - **室内场景：** 通过 SceneSmith 生成新室内环境或从其场景库抽取，并在 Genesis 中进行物理仿真。
  - **人体建模：** SMPL-X 参数化模型 $M(\theta,\beta)$；先由 LLM 将姿态与体型描述扩展为语义描述，再通过 few-shot 示例对映射为关节角；构建 URDF 并用 SMPLitex 纹理化（仅支持坐姿与卧姿，卧姿通过物理落位更新参数）。
  - **VLM/LLM 底座：** 全流程（Generator、Critic、Orchestrator）默认使用 Gemini 3.7 Flash（论文指出为可配置管线参数）。
- **四阶段 Generator-Critic 架构（§3.1 & Table 4）：**
  1. **Human Generation：** 输出体型 $\beta$、髋/膝/颈/肩/肘关节角及坐/卧姿态类别；Critic 检查双视角渲染，带自碰撞对数与最大穿透深度（$\ge 1\text{ cm}$）咨询旗标。
  2. **Human Placement：** 基于空家具俯视图输出归一化 $(x,y)\in[0,1]^2$ 与 `head`/`facing` 六向离散方向；Critic 结合俯视与等轴测图检查穿模与靠边可达性。
  3. **Robot Placement：** 生成 `get_robot_placement(human)` Python 代码，仅能调用 11 个人体关键点 API `get_joint_pos` 返回底座位置与 yaw；门控为落位后底座倾角 $\le 1^\circ$。
  4. **Robot Motion Generation：** 生成 `generate_trajectory(...)` Python 代码，可调用 12 个关键点及点云表面投影、法向与相机位姿 API，输出路点序列（位置、$3\times 3$ 姿态、速度、力标志 $\in\{0,1\}$、规划器 $\in\{\text{RRT},\text{Cartesian}\}$）；门控包括 schema 断言（接触速度 $<0.05\text{ m/s}$、首路点无力）及全路点可达，Critic 额外审查执行视频。
- **Orchestrator 跨阶段协调（§3.2 & Table 5）：**
  - 任务扩展生成不可变 $\tau$；Critic 仅依据原始任务和 $\tau$ 评审，防止任务漂移（task drift）。
  - 支持三类动作：`accept`、`steer`（重开或续跑当前循环并注入指令）、`backtrack`（按依赖图使下游资产失效并回溯上游重算）。
  - 策略学习分支扩展至数据采集、高层/低层策略训练与仿真评测三阶段（Table 5 列出对应门控与咨询旗标）。
- **双路径真机迁移（§3.3 & Appendix C–E）：**
  - **Direct Execution（免训练直接执行）：** 单目 RGB 经 Fast SAM3D Body 拟合 SMPL-X 网格并开环执行生成轨迹；在窄小部位（前臂宽 $5\text{--}8\text{ cm}$）存在 $6\text{--}16\text{ cm}$ 深度误差（Appendix C），因此未用于真机挠痒评测，仅用于擦浴快速预览。
  - **Policy Learning（分层模仿学习）：** 基于 ArticuBot 改进，高层将骨架主干从 PointNet++ 替换为 Point Transformer V3 并加入离散路点索引条件 $I\in[0,K-1]$（当末端 4 点均值距离低于 $d_{\text{thresh}}$ 时递增）以解决往复接触的多模态歧义；低层为 8 步动作预测视界的 3D Diffusion Policy。
  - **数据与训练规模：** 每任务在 100 个人体上采集 1,000 条示范（点云下采样至 4,500 点），保留原轨迹并做 16 倍轨迹扰动增强 + 在线观测噪声；4 张 NVIDIA RTX PRO 5000 Blackwell (48 GB) 上单任务训练约 19–20 小时（擦浴 19.8 h、挠痒 19.0 h）。
- **关键实验数据（§4–§5 & Table 1–3, 6–7）：**
  - **50 个场景规模化生成（Table 1, 6）：** 覆盖 9 个护理任务族（Back percussion 7、Itch relief 7、Sponge bathing 6、Lotion application 5、Skin assessment 5、Hand warming 5、Towel drying 5、Comforting touch 5、Edema massage 5）、34 种房间布局；总耗时 254.4 GPU-hours、9,170 次 LLM 调用（9.17M 输入 / 605k 输出 tokens）、1,209 次阶段尝试；50/50 场景至少触发 1 次 Critic 拒绝（平均 16.0 轮），16 个场景触发 Orchestrator 共 55 次干预（42 次带上下文重试、5 次局部 steer、3 次 backtrack、5 次 accept 历史最优）。
  - **7 路消融实验（Table 2，Back-and-leg bathing，每组 100 个人体）：** Full pipeline 全路点可达率 100.0%、轨迹偏移 $>2\text{ cm}$ 占比 0.0%、P95 末端误差 1.8 mm；去掉 Orchestrator（Critics only）降至 57.0% / 4.0% / 152.3 mm；再分别移除单阶段 Critic 后降至 0%–37%；Open-loop 为 0.0% / 100.0% / 349.3 mm。
  - **真机与仿真用户实验（Table 3，$N=12$ 被试，Hello Robot Stretch 3 + 天花板 ZED 2i + SAM 3 分割）：** Bathing 真实完成度 Policy 80±11% vs Direct 39±25%（$p<0.001$），仿真 Policy 86±7%；Scratching 真实 Policy（静态）79±16%、Policy+Motion（受试者主动移臂）63±20%，仿真分别为 92±14% 与 92±20%。提示词一致性（7 分制中位数）：仿真 5.5–6.5，真机 Bathing Policy 5.0 / Direct 4.5，真机 Scratching Policy 4.25。
- **原文局限（§6 Limitations）：** 真机仅在 12 名受试者、单台 Stretch 3 上验证 2 个任务；单视角遮挡与人类主动运动导致移臂条件下成功率降至 63%（训练中未加入环境人体运动）；当前采用刚体近似且策略未使用力传感器观测。
