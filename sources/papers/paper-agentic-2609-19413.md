# From Rollout to Reset: A Graph-Based Harness for Autonomous Long-Horizon Manipulation Evaluation

- **作者：** Jing Jiang*, Yue Yang*, Xinkai Jiang, Gedas Bertasius, Daniel J. Szafir, Rudolf Lioutikov†
- **机构：** Karlsruhe Institute of Technology (KIT), University of North Carolina at Chapel Hill (UNC-Chapel Hill)
- **版本 / 载体：** arXiv:2609.19413v1 [cs.RO]（Submitted on 16 Sep 2026，arXiv HTML 全文 + 官方项目页）
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2609.19413)、[arXiv HTML 全文](https://arxiv.org/html/2609.19413)、[官方项目页](https://yy-gx.github.io/HALTER/)
- **项目页 / 代码：** 宣称将开源或未列可运行代码（截至 2026-11-27，项目页链接 [YY-GX/HALTER](https://github.com/YY-GX/HALTER) 存在但仅含 `README.md`，注明 `Code release is in progress.`，尚无公开源码或权重文件）
- **核对范围：** arXiv HTML 全文全部章节（Abstract、§I Introduction、§II Related Work、§III Methodology A–D、§IV Experiments A–E、§V Conclusion、References）、Table I–III、Fig. 1–4 题注，以及官方项目页（HTTP 200）与 GitHub 仓库 `YY-GX/HALTER` README 预览。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [From Rollout to Reset: A Graph-Based Harness for Autonomous Long-Horizon Manipulation Evaluation（HALTER）](../../wiki/entities/paper-agentic-2609-19413.md)
- **研究定位：** 面向真机长程操作策略的自动化评测与场景复位 harness（简称 HALTER：*Harness for Autonomous Long-horizon Task Evaluation and Reset*），针对单步自动复位方法（如 AutoEval）在长程任务中因终态组合爆炸与局部遮挡而失效的问题，提出「在线时空场景图 + 原子复位技能库分层规划」方案。
- **问题形式化（§III-A）：**
  - 任务由自然语言指令 $x$ 指定，分解为原子技能集合 $\mathcal{S}=\{a_1,\dots,a_H\}$ 及因果偏序 $\prec$，合法执行序为 $\mathrm{Lin}(\mathcal{S},\prec)$。
  - 初始条件由物体关系集合 $\mathcal{C}_0=\{r_j(o_j,o_j')\}$ 约束，而非刚性 6D 位姿。
  - 核心双难点：(P1) 终态覆盖度随任务步数 $H$ 组合增长；(P2) 需在同一次 rollout 中完成三类场景状态估计——(A) 完成度评分 $S_{\mathrm{comp}}\in[0,1]$、(B) 复位规划、(C) 复位后终态核验。
- **核心架构（§III-B–D, §IV-B）：**
  - **在线场景图（0.5 Hz）：** 从三路 RGB-D 相机（2 台 DepthAI OAK-D Lite + 1 台 ZED）采样关键帧，用 GroundedSAM（Grounding DINO + SAM）按任务物体名称做开放词表实例分割，反投影到对齐深度图得到物体级点云与世界系包围盒，并用几何规则函数标注 `on`、`in`、`under` 与抽屉 $y$ 轴开合状态；物体被遮挡时保留最近有效观测与关系，重现时更新。
  - **统一 LLM 推理（Qwen3.5-397B）：** 将场景图历史序列化为物体记录与关系三元组，通过阶段专用提示词完成三项推理：历史轨迹完成度评分 $\widehat{S}_{\mathrm{comp}}$、基于原子技能库 $\mathcal{K}$ 的复位序列规划 $P=(k_1(\alpha_1),\dots,k_N(\alpha_N))$（技能参数绑定图中已有物体节点）、以及两级复位校验（每步后效校验 + 终态对 $\mathcal{C}_0$ 核验，失败则中止并请求人工介入）。
  - **原子复位技能库：** 两个桌面场景共 12 项原子复位技能（抽屉场景 6 项、厨房炉灶场景 6 项），每项使用 30–50 条演示训练 X-VLA 策略；提示词每任务附 4 条参考样例（1 成功 + 3 失败）。
- **核心实验数据（§IV，Franka Panda + Robotiq 2F-85，被测策略为 BESO，每任务 25 episode）：**
  - **4 个主任务（Drawer-Store / Drawer-Retrieve / Pot-Cook / Pan-Cook，共 100 episode，Table I & II）：** HALTER 复位成功率 **76.0 ± 4.3%**（AutoEval 52.0 ± 5.0%，MP skills 65.0 ± 4.8%），评分准确率 **90.0 ± 3.0%**（AutoEval 76.0 ± 4.3%），规划有效率 **84.0 ± 3.7%**，复位核验准确率 **91.0 ± 2.9%**（AutoEval 78.0 ± 4.1%）；人工介入从 100 次降至 25 次，操作员耗时从 88 min 降至 24 min（减少 **72%**），单回合周期 121 s（AutoEval 104 s，MP skills 137 s，人工 53 s）；复位数据量 105k 步（较 AutoEval 138k 步少 24%）。Pot-Cook 上 MP skills（72%）高于 HALTER（48%），系 X-VLA 对汤锅抓取稳定性不及解析抓取。
  - **3 个留出任务组合泛化（Store / Retrieve / Pot，零新增演示，Table III）：** HALTER 复位成功率 **74.7 ± 5.0%**、评分准确率 **94.7 ± 2.6%**、规划有效率 **89.3 ± 3.6%**、核验准确率 **92.0 ± 3.1%**；AutoEval 复位成功率仅 **1.3 ± 1.3%**（0% / 4% / 0%）。
  - **场景图消融（Drawer-Store，25 episode/组，Fig. 4）：** 用原始图像替代场景图使规划有效率 92%→72%、核验准确率 96%→76%、复位 SR 88%→72%；用渲染点云替代则进一步降至 40% / 52% / 40%。更新率在 0.2 Hz 仍可维持性能，低于 0.2 Hz 时规划有效率降至 64%、核验降至 60%、复位 SR 降至 48%（评分仍为 96%）。
- **局限（§V & 项目页）：** (1) 技能库须覆盖正向任务未出现的逆向复位技能（仍需演示）；(2) 每个新任务仍需编写 4 条参考上下文样例；(3) 物理不可逆状态（如物体破损、液体洒出）无法通过技能规划恢复，仅能触发核验报警转交人工；(4) 复位执行本身仍受长程复合误差制约。
