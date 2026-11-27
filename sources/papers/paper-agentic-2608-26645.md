# FLARE: A Failure-Aware Framework for Autonomous Correction and Recovery in Visual-Language Robotic Manipulation

- **作者：** Ganlong Zhao, Zijia Tang, Xingping Chen, Zhanghui Kuang, Ye Tian, Guanbin Li（通讯作者）
- **机构：** The Chinese University of Hong Kong（香港中文大学）、Centre for Perceptual and Interactive Intelligence、Duke University（杜克大学）、Sun Yat-sen University（中山大学）、TengenX、Tencent Robotics X（腾讯）、Shenzhen Loop Area Institute（深圳河套研究院）、Guangdong Key Laboratory of Big Data Analysis and Processing
- **版本 / 载体：** arXiv:2608.26645v1 [cs.RO]（2026-08-27 提交）；Accepted to CVPR 2026
- **原文：** <https://arxiv.org/html/2608.26645>（arXiv HTML 全文）；摘要页：<https://arxiv.org/abs/2608.26645>
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv 原文、HTML 链接扫描与元数据补查均未发现独立项目页或官方 GitHub 仓库链接）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§1–§6、Table 1–5、Fig. 1–5 说明、References，以及 Supplementary Material §1–§4、Table 6–7、Fig. 6–7 说明）
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [FLARE（paper-agentic-2608-26645）](../../wiki/entities/paper-agentic-2608-26645.md)

### 1. 问题定义与 ID/OOD 失败分类学

- 论文指出，当前视觉-语言-动作模型（VLA，如 $\pi_0$、$\pi_{0.5}$、OpenVLA）脆弱性的深层原因在于示范数据呈**轨迹单调（trajectory-monotonic）**与**成功偏置（success-centric）**分布：策略 $\mathbf{a}_t \sim \pi_\theta(\cdot \mid o_t, \mathcal{I})$ 将环境状态 $s_t^e$ 与机器人自身状态 $s_t^r$ 强耦合，导致条件分布 $P(s_t^r \mid s_t^e, \mathcal{D}_{\text{demo}})$ 方差极低。
- **分布内误差（ID Error）：** $s_t^e \in \mathcal{S}_{\text{task}}^e$（环境仍处于有效任务轨道），但 $s_t^r$ 落入示范条件分布的低概率区域（如抓空后末端处于放置位姿）。仅需通过 **Retry** 从新位姿重新执行下一步。
- **分布外误差（OOD Error）：** $s_t^e \notin \mathcal{S}_{\text{task}}^e$（环境状态已被破坏，如杯子倾倒、咖啡胶囊卡槽），任务策略无法直接挽救，需要专用 **Reset** 技能将环境恢复到合法状态。

### 2. 核心方法细节

1. **Perturbation & Bridging（实现内置 Retry）：**
   - 先沿用 MimicGen 从少量人类示范（每任务 10 条）拆分并拼接子任务序列 $\mathcal{T}_{\text{task}} = (a_0, a_1, \dots, a_N)$，扩增至 500 条。
   - 在起点和各子任务交界处注入扰动-桥接片段 $d_i = (d_i^A, d_i^B)$：扰动阶段 $d_i^A$ 施加随机运动（仿真默认最大旋转 $45^\circ$、最大平移 $0.5\text{ m}$）以打破位姿-状态耦合；桥接阶段 $d_i^B$ 采用无规划器运动学插值（位置线性插值 + 姿态 SLERP）回到下一子任务 $a_{i+1}$ 的合法起始位姿。
   - 仅将原始成功子序列与 $(d_i^B, a_{i+1}, \dots)$ 纳入训练监督，丢弃无意义的随机扰动动作 $d_i^A$。
2. **MLLM 驱动的失败挖掘与 Object-Centric Reset 技能学习：**
   - 运行具备 Retry 能力的模型采集数百回合视频与轨迹日志，使用离线 MLLM（Gemini-2.5-Pro，采样温度 $0.7$）解析失败视频并输出结构化 JSON（`error_type`、`reset_target`、`error_group`、`failure_timestamp`、`confidence`）。
   - 对需要多物体联动复位的故障（如胶囊被咖啡机盖压住），Prompt 强制按操作依赖选择先操控的对象（先开盖而非直接抓胶囊），并将 `error_group` 作为整体语义失败资产保持相对位姿进行场景布局扩增。
   - 依据时间戳提取失败状态（仿真为 MuJoCo 状态；真机通过 Any6D 估计物体 6D 位姿近似复现，无需精确坐标匹配），针对每个 reset 目标采集少量人类示范（仿真 20 条增广至 500 条；真机 10 条增广至 50 条）并叠加相同的 Perturbation & Bridging。
3. **模块化 LoRA 技能库与闭环仲裁：**
   - 底座采用 $\pi_{0.5}$（冻结除语言模型与动作专家外的参数），为任务主策略 $\pi_{\text{LoRA}}^{\text{task}}$ 和各对象级重置策略 $\pi_{\text{LoRA}}^{\text{reset}, j}$ 分别训练独立 LoRA 适配器（Adam，恒定学习率 $2.5 \times 10^{-4}$，无衰减）。
   - 在线推理时，MLLM 监控器识别到 OOD 错误后下发对应重置指令并切换至 $\pi_{\text{LoRA}}^{\text{reset}, j}$，环境恢复后再切回 $\pi_{\text{LoRA}}^{\text{task}}$。

### 3. 核对的关键实验数据与局限

- **仿真（RoboMimic 9 任务，每任务 50 次试验，Table 1）：**
  - 平均成功率：OpenVLA `38.0%`，Task-conditioned `41.8%`，Subgoal-conditioned `43.8%`，Motion-conditioned `46.9%`，Subgoal Self-reflection `48.0%`，Phoenix `57.8%`，Phoenix-Human `78.9%`，$\pi_{0.5}$ `72.2%`，FLARE（Ours）`84.0%`（相对底座 $\pi_{0.5}$ 提升 `+11.8%`，在 9 项任务中 8 项最高，Threading_D0 为 `72%` 低于 Phoenix-Human 的 `100%`）。
- **超参敏感性（Coffee_D1，Fig. 4）：** 单因素测试中旋转 $r = 30^\circ$ 与平移 $t = 0.7\text{ m}$ 分别取得峰值表现，过大扰动会降低示范生成效率并因训练方差过高而退化。
- **消融与模块评测（Table 3–5）：**
  - 移除 Reset（`Ours w/o Reset`）使 Coffee / ThreePieceAssembly 平均成功率下降 `3.5%`；人工指令上界（`Ours-Oracle`）再提升 `7%`。
  - 对象级重置单技能成功率（Table 4）：Coffee 机盖 `84%`、咖啡胶囊 `24%`（生成效率 `83.7%` vs `11.6%`）；ThreePieceAssembly T 型块 `88%`、U 型块 `20%`（生成效率 `48.6%` vs `5.9%`）。
  - Gemini-2.5-Pro 离线视频识别（50 条人工标注视频，Table 5）：Coffee 任务 Retry/Reset 分类 `88%`、目标对象 `88%`、时间戳 `78%`；ThreePieceAssembly 分别为 `96%`、`78%`、`66%`。
- **真机实验（Piper 机械臂 + 双目/顶视与腕部 RealSense D435i，每任务 40 次试验，Table 2）：**
  - Stack Three Blocks：$\pi_{0.5}$ `62.5%` vs FLARE `75.0%`；Insert U-shaped Block：$\pi_{0.5}$ `45.0%` vs FLARE `55.0%`。
- **原文明确指出的局限：**
  - 无法恢复超出夹爪尺寸或工作空间范围的故障（如整个咖啡机翻倒、物体掉落至地面）；
  - 需要预先定义可重置对象列表并采集示范，尚不能零样本生成全新多步协同重置动作；
  - 精细位姿调整（如扶正倒下的胶囊或 U 型块）受限于硬件末端灵活性与手内调整能力，重置成功率仅 `20%–24%`。
