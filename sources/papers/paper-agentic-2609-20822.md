# SafeHarness: Coding Agents with Harness for Safe Robot Control

- **作者：** Bingxin Xu, Yuzhang Shang, Zhen Dong, Emilio Ferrara
- **机构：** University of Southern California (USC); University of Central Florida (UCF); University of California, Santa Barbara (UCSB)
- **版本 / 载体：** arXiv:2609.20822v2 [cs.RO]（2026-09-17 提交 v1，2026-10-02 修订 v2；HTML 版主标题为 *SafeHarness: Coding Agents with Harness for Safe Robot Control*，arXiv 摘要页标题为 *Coding Agents with Harness for Safe Robot Control*）
- **原文：** [arXiv HTML v2](https://arxiv.org/html/2609.20822) / [arXiv Abs](https://arxiv.org/abs/2609.20822)
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv:2609.20822v2 正文、脚注与参考文献均未列出 GitHub 仓库、模型权重或独立项目页 URL；`context.json` 中亦无官方候选代码仓库）
- **核对范围：** arXiv:2609.20822v2 HTML 全文（Abstract、§1 Introduction、§2 Related Work、§3.1–§3.4 Method、§4.1–§4.5 Experiments、§5 Conclusions、Fig. 1–4 图注、Tab. 1–4 全部数据与 References [1]–[53]）
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **核心问题与失败诊断（§1, §3.1, §3.4, §4.3）：**
  - 既有面向机器人操作的编码智能体（如 Code as Policies、Harness VLA 等）仅以「任务目标是否达成」评测执行结果，未考核过程碰撞安全性。
  - 在 SafeLIBERO（每项任务配对一个禁碰障碍物）评测中，即便提示词明确禁止触碰障碍物、且推理轨迹（traces）已正确识别障碍物，原始编码智能体仍以最短路径直穿障碍物或直接回放无障碍场景积累的技能记忆（`Model + Skills` 在 Level I 上碰撞率显著恶化），原因在于**安全约束从未进入规划优先机制**，且在长上下文（数百步交互、数万 token）下仅靠 prompt 维持安全约束极易衰减。
- **阶段切分与 SafeHarness 双护栏架构（§3.1–§3.3）：**
  - 按「接触事件（contact event）」将操作任务切分为若干子阶段，每阶段由「自由空间路径段」与「末端富接触（contact-rich）动作段」组成，分别施加两类互不替代的障碍物感知护栏：
  1. **Obstacle-Aware Route Planning（§3.2）：**
     - 使用分割模型（SAM 3）将目标物体与障碍物定位为三维/场景边界框（bounding boxes）。
     - 编码智能体先生成以路点折线（polyline of waypoints）表示的候选路径，并交由**对话上下文之外的外部验证器（verifier）**检查路径是否侵入障碍物盒（水平/一般安全裕度 $12\text{ cm}$，上方净空裕度 $5\text{ cm}$；抓持物体后将夹爪与被抓物体视作整体共同校核）。
     - **两点触发重规划（Two-point replanning）：**（a）执行前验证不通过即驳回重规划；（b）执行途中若出现未预见的接近风险或夹爪在距离目标路点超过 $3\text{ cm}$ 处受阻停滞（stall），立即停机并从当前实际位姿重新规划。
     - **观测经济性（Economy of observation）：** 限制执行阶段重复读取同一路径规划图像的次数，防止相同图像反复追加进上下文窗口引发模型退化。
  2. **Obstacle-Aware Contact Execution（§3.3）：**
     - 路径终结于目标上方的预抓取（pre-grasp）或释放（release）位姿；下方开爪、下探与闭合动作会扫掠路径验证未覆盖的局部体积。
     - 对目标与障碍物做邻近检测（adjacency test，夹爪接触足迹距障碍物 $< 2\text{ cm}$ 时触发）：若无邻近障碍物，采用默认接触策略；若有邻近障碍物，优先选择与障碍物保持安全净空且距离最远的接近方向；若所有方向均不满足净空，则旋转夹爪使张开轴（opening axis）切向平行于障碍物并垂直下放负载，避免横向扫掠碰撞；夹爪进入目标 $3\text{ cm}$ 范围内才交由冻结的 VLA 原语完成精细接触。
- **实验验证与核心指标（§4.1–§4.4，Tab. 1–4）：**
  - **基准与指标：** SafeLIBERO 共 16 个基础任务（Spatial / Goal / Object / Libero-10 各 4 个）$\times$ 2 难度等级（Level I 障碍紧邻目标，Level II 障碍位于搬运路径上）= 32 任务 $\times$ 10 随机种子（每级 160 episodes）；障碍物位移 $> 1\text{ mm}$ 即判为碰撞。指标为任务成功率 TSR、避障率 CAR 与安全成功率 SSR（同时成功且无碰撞）。每集最多 900 步（成功平均约 400 步，双子目标 Libero-10 约 700 步）。
  - **SafeLIBERO 主结果（Tab. 1）：**
    - `SafeHarness (GPT-6-Astra)` 平均 **TSR 81.2% / CAR 91.9%**（Spatial 76.2 / 90.0，Goal 78.8 / 90.0，Object 92.5 / 91.2，Libero-10 77.5 / 96.2）。
    - `SafeHarness (GPT-5.5)` 平均 **TSR 78.8% / CAR 86.9%**（Spatial 81.2 / 76.2，Goal 68.8 / 88.8，Object 90.0 / 87.5，Libero-10 75.0 / 95.0）。
    - 对比基线：OpenVLA-OFT 平均 26.2 / 11.3；$\pi_{0.5}$ 平均 57.8 / 17.1；AEGIS 平均 67.5 / 68.9（在 Libero-10 上 AEGIS 的 TSR 46.3 甚至低于其包裹的冻结 $\pi_{0.5}$ 54.3）。
  - **真机实验（Tab. 2）：** PiPER-X 6-DoF 机械臂，2 任务（White can $\to$ plate、Coke bottle $\to$ plate）各 20 次试验（场景同时含目标旁与路径上 2 个障碍物）：`SafeHarness (GPT-5.5)` 平均 **TSR 65.0% / CAR 82.5%**（White can 70 / 85，Coke bottle 60 / 80），显著超越同底座无护栏 `GPT-5.5`（TSR 27.5% / CAR 45.0%）与 $\pi_{0.5}$（TSR 25.0% / CAR 17.5%）。
  - **消融实验（Tab. 3 & Tab. 4）：**
    - Tab. 3（Model vs. Model+Skills vs. SafeHarness）：`GPT-6-Astra` 下 `Model + Skills` 总体仅 50.0 / 34.4 / 18.8（TSR/CAR/SSR），加入 SafeHarness 提升至 **81.2 / 91.9 / 78.1**（SSR 提升 +315%）；`GPT-5.5` 下由 56.2 / 43.8 / 31.3 提升至 **78.8 / 86.9 / 73.1**（SSR 提升 +134%）。
    - Tab. 4（组件消融）：两个护栏缺一不可；路径规划贡献大于接触执行（如 GPT-6-Astra 下 `MS` 18.8 $\to$ `MS + Contact` 39.4 $\to$ `MS + Route` 64.4 $\to$ `SafeHarness` 78.1 SSR）。
- **对 wiki 的映射：** [SafeHarness: Coding Agents with Harness for Safe Robot Control](../../wiki/entities/paper-agentic-2609-20822.md)
