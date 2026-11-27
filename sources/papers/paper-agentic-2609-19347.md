# Kinematics-Grounded Agentic AI for Robotic Additive Manufacturing Process Planning

- **作者：** Jingzhan Ge, Ruimin Chen, Azadeh Haghighi, Jiong Tang, Farhad Imani
- **机构：** University of Connecticut（School of Mechanical, Aerospace, and Manufacturing Engineering）；University of Illinois Chicago（Department of Mechanical and Industrial Engineering）
- **版本 / 载体：** arXiv:2609.19347v1 [cs.RO]（2026-09-16 提交，25 页，17 张图，CC BY 4.0）
- **原文：** [arXiv Abstract](https://arxiv.org/abs/2609.19347) · [arXiv HTML 全文](https://arxiv.org/html/2609.19347)
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv 原文及元数据未列出独立项目页或官方 GitHub 仓库链接）
- **核对范围：** 已逐段通读 arXiv HTML 全文（Abstract、§1 Introduction、§2 Related Work 2.1–2.4、§3 Methodology 3.1–3.8 与公式 (1)–(34)、§4 Experimental Design and Results 4.1–4.3 与 Table 1–7 及 Fig. 1–17 图注、§5 Conclusion 与 References [1]–[45]）
- **通读：** 已完成

## 原文核对笔记

- **核心问题与定位：** 提出面向机器人增材制造（Robotic AM / FFF）的智能体—专家—工具规划框架 **A-RAM**（Agentic Robotic Additive Manufacturing），解决切片级工艺参数（填充图案、密度、角度、层配置）、零件平面朝向（part orientation）与工作空间摆放位置（workspace placement）在机械臂逆运动学（IK）、轨迹时间参数化、关节空间冲击（Joint-6 jerk）与同步挤出之间强耦合的问题。
- **架构职责分离：**
  - **Triage Agent（LLM，采用无任务微调的 Qwen3-8B 与确定性解码）：** 仅负责理解自然语言制造意图、约束与固定/待搜寻变量，输出受 Schema 约束的规划请求 $\rho(u) = (\omega, \boldsymbol{\kappa}, \mathcal{V}_{\mathrm{fix}}, \mathcal{V}_{\mathrm{search}}, \mu)$；遇到无效字段触发有界修复提示或交由人工澄清，不直接预测数值指标。
  - **Planning Agent（确定性）：** 将 $\rho(u)$ 映射到查询条件化的搜索范围 $\sigma(u)$（全交叉搜索 $\sigma_{\mathrm{full}}$、两阶段筛选 $\mathcal{Z}_{\mathrm{screen}}$ 或固定工艺的朝向—摆放搜索 $\sigma_{\mathrm{place}}$）以及评估配置档 $\eta$（区分筛选替代档 $\eta_{\mathrm{pat}}/\eta_{\mathrm{plc}}$ 与最终打印档 $\eta_{\mathrm{F}}$）。
  - **领域专家与工具链：** CAD/摆放工具（PrusaSlicer 前置几何投影筛查 + $160\text{ mm} \times 160\text{ mm}$ 含 $2\text{ mm}$ 安全裕度构建区上的 C1–C3 中心采样）、切片工具（PrusaSlicer CLI）、运动规划专家（恒加速度前瞻速度规划 + 多初值顺序 IK + $\delta q_j^{\max}=0.35\text{ rad}$ 连续性门控 + Joint-6 周期解卷绕 + 三次 B 样条 $\Delta t_c=0.005\text{ s}$ 重采样 + 3 次二阶中心差分求 $j_6^{\max}, \bar{j}_6$）、挤出专家（按局部 TCP 位移比例将切片挤出量 $e_s$ 重新分配到机器人轨迹时间步）与执行/数字影子层（ROS 2 / MoveIt 模拟分支与 UR3 RTDE 实机打印与命令侧监测）。
- **实验验证要点（UR3 六轴机械臂 FFF 单元，$h=1\text{ mm}$、挤出宽 $2.5\text{ mm}$、$v_{\max}=25\text{ mm/s}$、$a_{\max}=d_{\max}=25\text{ mm/s}^2$）：**
  - **Case Study I（Dogbone Type IV）：** 专家指定参数任务（I-a，$1\times 1\times 3$）中 `45°/C3` 较 `45°/C1` 将 $j_6^{\max}$ 降低 10.4%、$\bar{j}_6$ 降低 18.3%；仅目标规划任务（I-b，$16\times 5\times 3=240$，239 有效、1 个 `hilbertcurve 90°/C2` IK 不可行）中 `stars, 30°/C3` 较最差有效方案 `honeycomb, 0°/C1` 分别降低 53.5% 与 48.3%。
  - **Case Study II（5 层 Husky Head，两阶段 $16+5\times 3=31$）：** 90% 填充率下，省料目标选中 `lightning`（$11049.11\text{ mm}$，比 `3dhoneycomb` 短 12.7%），最短时间目标选中 `supportcubic`（$1131.7031\text{ s}$，比 `hilbertcurve` 快 35.5%）；100% 填充率下仅 5 种图案可切片，省料选中 `octagramspiral`（$11993.30\text{ mm}$，降 5.1%），最短时间选中 `rectilinear`（$1137.91\text{ s}$，降 40.1%）；零填充含实体层阶段统一选中 `90°/C3`（$j_6^{\max}=563.067\text{ rad\,s}^{-3}$，降 29.5%）。
  - **Case Study III（仅轮廓运动质量查询，$1\times 5\times 3=15$）：** 4 层 Gator Head 选 `0°/C3`（$j_6^{\max}$ 降 31.1%），4 层 Husky Head 经 1% 近并列字典序规则选 `0°/C1`（$j_6^{\max}$ 降 22.6%），5 层 Husky Head 选 `90°/C3`（$j_6^{\max}$ 降 30.6%），证实最优朝向与摆放位置随零件几何及层数变化。
- **局限边界：** 固定平面喷嘴姿态；运动可行性仅覆盖采样点运动学与关节连续性，未包含完整动力学与全局碰撞检测；Joint-6 jerk 仅反映局部腕部运动而非整臂动力学或成形件力学性能；数字影子挤出宽度为命令侧体积守恒代理值而非编码器/视觉实测闭环；开源代码与项目页尚未公开。

对 wiki 的映射：[Kinematics-Grounded Agentic AI for Robotic Additive Manufacturing Process Planning](../../wiki/entities/paper-agentic-2609-19347.md)
