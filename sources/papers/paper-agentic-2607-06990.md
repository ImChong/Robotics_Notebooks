# A Closed-Loop Multi-Agent Framework for Robust Multi-Robot Manipulation

- **作者：** Yi-Xiang He, Lan Wei, Haoming Cen, Jian-Jian Jiang, Zhuohao Li, Guanxing Lu, Yihan Yang, Dandan Zhang, Wei-Shi Zheng
- **机构：** Sun Yat-sen University; Key Laboratory of Machine Intelligence and Advanced Computing, Ministry of Education, China; Imperial College London; Tsinghua Shenzhen International Graduate School; Shenzhen Loop Area Institute
- **版本 / 载体：** arXiv:2607.06990v1 [cs.RO]（Submitted on 8 Jul 2026; Comments: RSS 2026）
- **原文：** <https://arxiv.org/html/2607.06990> / <https://arxiv.org/abs/2607.06990>
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27 核查：arXiv HTML 全文、摘要页及候选仓库索引均未列出独立项目主页或官方代码仓库链接；系统调用第三方商业 API 与已有开源组件如 Qwen3-Max、Qwen3-VL-Plus、SAM、AnyGrasp、PyRoki、FAST-LIO）
- **核对范围：** 已通读 `/tmp/agentic_172_cache/paper-agentic-2607-06990/paper_text.txt`（arXiv HTML 全文 833 行，79,275 字符），完整核对 Abstract、Section I–V、Table I–V、Fig. 1–8 图注、References 及 Appendix A（硬件平台）、Appendix B（感知与抓取/旋转工具细节，含 Eq. 1–2）、Appendix C（模型快照版本与简化系统提示词）。未本地复现或运行真机实验。
- **通读：** 已完成

## 原文核对笔记

1. **核心问题与架构定位：**
   - 现有 LLM 机器人工作分两类：单臂/单机操作方法（如 Code as Policies、VoxPoser、ReKep）缺乏跨工作空间协同机制；多机协同规划方法（如 RoCo、SMART-LLM、Coherent）多停留在开环高层分配，将底层操作视为理想化原语，遇到抓取滑移、外力扰动或可达性不足时容易引发级联失败。
   - 本文提出由 **Planning Agent（规划）— Manipulation Agent（执行）— Verification Agent（验证与分级恢复）** 构成的分层闭环多智能体框架，在异构多机系统（移动双臂 Agilex Cobot Magic、固定单臂 Agilex Piper、桌面双臂 ABB YuMi）上落地。

2. **三大智能体职责与关键机制（Section III & Appendix B–C）：**
   - **Planning Agent（Qwen3-Max, `qwen3-max-2026-01-23`）：**
     - 结合自然语言指令、粗粒度场景描述与异构机器人能力，输出带 `parallel` 标志的子任务 DAG（JSON）；
     - 指令或场景存在歧义时发出 `further_perception` / `further_allocation` 标志向对应 Manipulation Agent 发起双向精细感知查询；
     - 接收 Verification Agent 上报的物理不可行错误后，保留已完成步骤并从失败步重规划（Global Failure Recovery）。
   - **Manipulation Agent（Qwen3-VL-Plus, `qwen3-vl-plus-2025-12-19`，每台机器人独立实例化）：**
     - **角色解析：** 区分主动/被动交互对象（Active/Passive）、语义部件描述及是否需要双臂（`dual_arms`）；
     - **工具增强视觉定位：** Qwen3-VL-Plus 输出 2D BBox → SAM 分割掩码 → 基于 ReSemAct 的网格视觉提示（Canny 轮廓增强 + ROI 归一化 + $10 \times 10$ 网格索引分类）→ $7 \times 7$ 邻域采样 + 离群值剔除 + 空间中值提取 3D 关键点；
     - **混合抓取与旋转推理：** 小物体（高度 $\le 0.05\text{ m}$）、关节物体或双臂对称抓取采用 BBox 几何启发式；一般物体调用 AnyGrasp 经运动学过滤后，在关键点 $\mathbf{p}_{key}$ 邻域 $r_k = r_{init} + k \cdot \Delta r$ 内迭代最大化置信度分数 $s_i$（Eq. 1–2）；旋转任务投影末端 X/Y/Z 轴到目标中心供 VLM 预测轴、方向与离散角度（$60^\circ/75^\circ/90^\circ/100^\circ$）；
     - **动作原语库（Table I，11 类）：** Basic Manipulation（`Grasp`, `Place`, `Lift Up`, `Put Down`, `Reset Home`）、Spatial Adjustment（`Move XY`, `Move Pose`, `Rotate`, `Align`）、Constraint Interaction（`Open`, `Close`），统一映射为末端位姿（EEF pose）并通过 PyRoki / YumiPy IK 执行；
     - **双层记忆：** 短期记忆保存交互历史以消解序列歧义（如“倒水”前是否已抓杯）；长期记忆经验池提取 `<Action> Active -> Passive` 模板签名，复现相同子任务签名时直接复用已验证拆解与原语代码以减少 VLM 调用。
   - **Verification Agent（Qwen3-VL-Plus，与 Manipulation Agent 一一配对）：**
     - 对比动作前后图像（`#Image_PRE#` 与 `#Image_POST#`）输出二元判定 JSON；
     - **内循环局部恢复（Local Recovery）：** 对可执行性未丧失的执行失败（Condition Loss 或 Ineffective Action），依据 `#HISTORY#` 生成精确匹配的 `replan_steps` 交回 Manipulation Agent 重试；
     - **外循环全局恢复（Global Recovery）：** 对物理不可行错误（如目标超出固定机械臂工作空间或物体缺失），上报错误上下文至 Planning Agent 触发跨机器人重新分配。

3. **实测实验结论（Section IV, Table II–V, Fig. 4–5）：**
   - **真机硬件：** Agilex Cobot Magic（双 6-DoF 臂 + 差分底盘 + Livox Mid-360 + FAST-LIO + ROS1 导航 + RealSense D435 + PyRoki）、Agilex Piper（6-DoF, 1.5 kg 负载 + D435 + PyRoki）、ABB YuMi（IRB 14000 + RealSense L515 + YumiPy）。
   - **桌面双臂任务（Table II，每任务 20 次试验；OpenVLA-OFT 与 $\pi_0$ 各用 50 条单任务演示微调；ReKep 提供人工真值关键点）：**
     - Block Stacking：OpenVLA-OFT `1/20` (35%)、$\pi_0$ `9/20` (46%)、ReKep Annot. `7/20` (56%)、本文 `14/20` (85%)；
     - Table Organization：OpenVLA-OFT `5/20` (55%)、$\pi_0$ `9/20` (75%)、ReKep Annot. `6/20` (44%)、本文 `11/20` (76%)；
     - Object Stowing（抽屉开关）：OpenVLA-OFT `4/20` (31%)、$\pi_0$ `8/20` (60%)、ReKep Annot. `0/20` (0%)、本文 `14/20` (78%)。
   - **外部扰动鲁棒性（Table III，每任务 10 次试验，覆盖 Action / Sub-task / Global 三级扰动）：**
     - OpenVLA-OFT 平均 `7%`（0/10, 0/10, 2/10）、$\pi_0$ 平均 `20%`（2/10, 1/10, 3/10）、本文平均 `63%`（5/10, 6/10, 8/10）。
   - **验证器分级指标（Table IV）：** 4 项任务失败识别率（ID）达 `90%–100%`，重规划成功率（Re-Plan）均达 `100%`，物理恢复率（Phy-Rec）`78%–100%`，最终任务成功率（Final SR）`40%–80%`。
   - **六任务消融与错误归因（Table V & IV-F）：** 去掉关键点感知工具（替换为 ReKep 特征聚类）退化最严重（6 项任务中有 5 项降至 `0/10`，Object Stowing 为 `2/10`）；去掉 Verification Agent 在长程 Table Org. 与 Food Prep. 上明显掉点（均降至 `3/10`）；去掉短期记忆在 Food Prep. 降至 `1/10`、Table Org. 降至 `2/10`；去掉经验池影响较小且引入 0 失败归因。全模型共 48 例失败中，关键点定位误差占比最高，其次为动作原语生成失败。

4. **对 wiki 的映射：**
   - [A Closed-Loop Multi-Agent Framework for Robust Multi-Robot Manipulation](../../wiki/entities/paper-agentic-2607-06990.md)
