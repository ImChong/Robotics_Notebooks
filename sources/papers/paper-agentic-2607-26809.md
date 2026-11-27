# Practice Makes Policies: Bootstrapping and Consolidating Robotic Capabilities from Zero Human Demonstrations

- **作者：** Jialiang Li, Yuhan Wang, Haojun Li, Gaojing Zhang, Yangtian Ye, Qipeng Liu, Haotian Liang, Wenzhao Lian
- **机构：** Shanghai Jiao Tong University；University of Sussex
- **版本 / 载体：** arXiv:2607.26809v1 [cs.RO]（2026-07-29 提交）
- **原文：** https://arxiv.org/abs/2607.26809（全文核对使用 arXiv HTML 版 https://arxiv.org/html/2607.26809）
- **项目页 / 代码：** https://hero-agent.github.io/（宣称将开源或未列链接：截至 2026-11-27，官方项目页标注 `Code (Coming Soon)`，尚未提供官方 GitHub 源码仓库 URL；论文仅提及使用第三方开源库 `huggingface/lerobot` 格式进行轨迹转换）
- **核对范围：** 通读 arXiv:2607.26809v1 全文（Abstract、Sections 1–6、Tables 1–3、Figures 1–6 题注及 Supplementary Material 全部小节含公式 (1)–(12)）及官方项目页 `https://hero-agent.github.io/` 文本内容。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[HERO（Practice Makes Policies）实体页](../../wiki/entities/paper-agentic-2607-26809.md)

1. **核心问题与总体架构（Sections 1 & 3）**
   - 开放世界机器人操作面临零样本泛化与高效闭环执行之间的权衡：基于基础模型的模块化智能体推理延迟高、缺少经验复用；端到端视觉-语言-动作策略（VLA）依赖大量人工演示且缺乏自主获取能力。
   - **HERO** 将机器人能力组织为三层能力谱系，并由基于 VLM（实验采用 `Gemini-3-Flash`）的层次化编排器（Orchestrator）统一调度两条闭环过程：
     - **H2E2R（自主能力演化，数据采集环）**：从零人工演示出发，经 L1 启发式推理（H）自举经验 $\to$ L2 范例迁移复用（E）加速采集，并由编排器合成逆向复位任务（reverse task）降低人工干预 $\to$ 将高频行为过滤并蒸馏为 L3 反射式闭环策略（R）。
     - **R2E2H（自适应任务执行，部署环）**：部署时优先调度 L3 闭环策略（R），在策略不可用或连续失败（每层最多 3 次重试）时回退至 L2 范例迁移（E），若仍无合适范例或迁移失败则回退至 L1 启发式推理（H），全过程结合执行监控与动态重规划（replanning）。

2. **三层能力与关键数学/工程细节（Section 3.1 & Supplementary Material）**
   - **L1：Heuristic Bootstrapper（启发式自举层）**
     - 多视角 RGB（3 个外部 RealSense D435i）经 **Depth-Anything-3 (DA3)** 重建 3D 工作空间；VLM 将语言子任务分解为两类、共 4 种原语技能并标注 2D 语义关键点：
       - 目标导向技能 $\mathcal{G}=\{t_{ee}, R_{ee}\}$：`Grasp`（3 个 2D 关键点 $p_l,p_r,p_h$ 提升至 3D 后按公式 (1) 构造正交末端坐标系并做轴向翻转修正）与 `Place`（1 个 2D 关键点确定 $t_{ee}$，继承前序朝向 $R_{ee}$ 后张开夹爪）。
       - 路径跟随技能 $\mathcal{T}=\{P_s, P_e, R_{ee}\}$：`Push`（起止 2 点对齐 TCP x 轴与推送方向，按 $x_{ee}\cdot x_{base}>0$ 消歧避免 IK 奇异）与 `Pull`（起止 2 点，继承前序抓持朝向）。
     - 后处理与过渡约束：末端工具中心点（TCP）偏置补偿 $t^*_{ee}=t_{ee}-0.15z_{ee}$（夹爪指尖距腕部 TCP 为 $0.15\text{ m}$）；当 Push/Pull 起止点深度差 $>5\text{ cm}$ 时启用射线-平面交点校正（公式 3）；技能前置接近与跨技能切换回撤均偏置 $d=0.15\text{ m}$（公式 5–6）。
   - **L2：Exemplar Accelerator（范例加速层，基于 MT3）**
     - **检索**：去停用词后以词级有序最长公共子序列（LCS）计算 F1 相似度 $s(a,b)=2\cdot\text{LCS}(a,b)/(|a|+|b|)$，仅保留 $s=1.0$ 且物体点云点数 $\ge 1000$ 的范例；维护质量分 $q\in(0.0,2.0]$（初始 $1.0$，复用成功 $+0.1$，失败 $-0.3$，$q\le 0.0$ 剔除）。
     - **运动迁移**：基于 **SAM 3** 掩码与 DA3 深度提取物体点云；粗配准用质心平移 + MT3 预训练 **PointNet++** 旋转回归器，精配准用多扰动（$\pm 2\text{ cm}$）**GICP**（最大对应距离 $0.10\text{ m}$、50 次迭代、$1.0\text{ s}$ 超时，内点 $<500$ 或 yaw 偏差 $>15^\circ$ 则回退粗配准）；采用 **4-DoF 归纳偏置**（剔除 roll/pitch，仅保留平移与 yaw，公式 8–10）。
   - **L3：Reflexive Policy（反射式策略层，基于 $\pi_{0.5}$-base）**
     - 输入为前视 + 腕部相机 RGB 与关节/夹爪状态（$15\text{ Hz}$），输出 50 步动作块（约 $3.3\text{ s}$），采用滚动时域执行前 30 步；从 $\pi_{0.5}$ 的 32 维输出取前 8 维 $[j_0,\dots,j_6,g]$ 做关节级控制。
     - **Action Expert 语言注入捷径**：针对标准 $\pi_{0.5}$ 中约 20 个文本 token 被约 512 个图像 token 稀释导致颜色指令不敏感的问题，新增线性投影层 $\mathbf{W}_{\text{lang}}\in\mathbb{R}^{2048\times 1024}$（约 $2.1\text{ M}$ 参数），经 $\text{Swish}$ 激活后广播加到全部 50 个动作 token（公式 11–12）。
     - **数据过滤与训练**：转为 LeRobot 格式，丢弃静止帧（$\|\mathbf{j}_{t+1}-\mathbf{j}_t\|_\infty < 0.001$），夹爪状态在 $0.5$ 阈值二值化并做 15 帧时间去抖；每任务约 158–180 条子任务回合（共 664 条），在 2 张 NVIDIA A100-SXM4-80GB 上用 AdamW 训练 $10\text{K}$ 步（batch size 32，约 5 小时）；推理时连续 8 步关节变化 $<0.02$ 或达 25 轮则终止。

3. **硬件、任务与核心实验数据（Section 4 & Table 1–3）**
   - **硬件与任务**：Franka Emika Panda 机械臂（附录提到 FR3 关节级控制）+ 4 台 RealSense D435i；4 项真机任务（各采集 30 条完整正/逆任务轨迹，累计 664 条子任务回合：Package Picking 180、Block Stacking 163、Drawer Search 158、Cuboid Discovery 163）。
   - **自主采集与演化（Q1）**：全任务平均每子任务人工干预仅 **$1.03\text{ s}$**，最长连续 **46** 个无人工干预子任务周期；逆向复位任务平均成功率 **84.3%**；编排器数据评估准确率为 **94.5%**；L1 与 L2 采集贡献占比分别为 **41.6%** 与 **58.4%**，有效采集速率分别为 **0.85** 与 **1.32** 成功子任务回合/分钟。
   - **自适应执行与消融（Q2，Table 1，每任务 30 次共 120 次试验）**：完整 HERO 平均成功率 **86.0%**（Package Picking **96.7%**、Block Stacking **66.7%**、Drawer Search **86.7%**、Cuboid Discovery **93.3%**），成功耗时均值 **3.9 min**；显著高于 `L1 only`（76.0%）、`L2 only`（66.7%）、`L3 only`（70.0%）及双层组合 `L1+L2`（73.3%）、`L1+L3`（82.7%）、`L2+L3`（72.5%）。关闭 Replan 后各单层成功率均明显下降（L1: $76.0\%\to 66.0\%$，L2: $66.7\%\to 55.0\%$，L3: $70.0\%\to 60.0\%$）。随机位移扰动下 L3 子任务成功率保持 **67.2%**。
   - **运行时统计与失败模式（Q3，Table 2 & Figure 6）**：部署中成功子任务由 L3 完成 **86.7%**、L2 回退 **6.5%**、L1 回退 **6.8%**；单子任务模块总耗时为编排器 **19.48 s**、L1 **46.57 s**、L2 **20.96 s**、L3 **37.90 s**；120 次任务中 73 次零失误完成，41 次触发恢复并成功挽回 30 次（最终 103 次成功），剩余失败含执行层 33 例（L3 策略执行错误 WPE 17、L2 物体分割 WOS 4 与运动迁移 WMT 4、L1 标注 WVA 3、L1/L2 3D 接地 WGD 4）及编排层 14 例（规划 WOP 12、监控 WOM 2）。
