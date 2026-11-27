# 2AM: Grounding Agent-Side Memory as Guidance for Steerable Action Models in Long-Horizon Manipulation

- **作者：** Yutong Hu, Fengjiao Chen, Xuezhi Cao, Renaud Detry（2026-11 项目页 AgentMem 修订版另列 Xiangming Zhu）
- **机构：** KU Leuven（Dept. Mechanical Engineering, Research unit Robotics, Automation and Mechatronics；Dept. Electrical Engineering, Research unit Processing Speech and Images）、美团（Meituan Inc.）、Flanders Make@KU Leuven（2026-11 AgentMem 修订版另列上海交通大学 Shanghai Jiao Tong University）
- **版本 / 载体：** arXiv:2609.11308v2 [cs.RO]（2026-09-10 提交 v1，2026-11-02 更新 v2）；项目主页于 2026-11 更新为 AgentMem 扩展版本
- **原文：** https://arxiv.org/abs/2609.11308 （HTML 全文：https://arxiv.org/html/2609.11308 ）；项目主页：http://agent-mem-action-model.github.io
- **项目页 / 代码：** 项目主页 http://agent-mem-action-model.github.io ；官方 GitHub 仓库 https://github.com/utomm/AgentMem-Action-Model （仓库以 AgentMem / `agent-mem-act` 发布，Hugging Face 模型与标注集合：https://huggingface.co/collections/you2who/agentmem-models-and-annotations-6b07deecb14d1c8e0abb5070 ；项目页源码仓：https://github.com/agent-mem-action-model/agent-mem-action-model.github.io ）—— 经 Step 2.5 核查判定为**已开源**
- **核对范围：** 通读 arXiv:2609.11308v2 HTML 全文（Abstract、§1 Introduction、§2.1–2.3 Contract between Agent and Action Model、Table 1、Fig. 1–3 题注、§3.1–3.3 Training-time supervision、公式 (1)–(7)、§4.1–4.2 Experiments、Table 2–3、§5 Related Work、§6 Discussion and Limitations、§7 Conclusion、References [1]–[31]）、项目主页文本快照（AgentMem 2026-11 修订版），并核查官方 GitHub 仓库目录树样本与 README
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [2AM（AgentMem）实体页](../../wiki/entities/paper-agentic-2609-11308.md)

### 1. 研究动机与部署约束（§1, §2.1）
- 现有长程操作两条主流路线存在归因混淆问题：
  1. 在 VLA 内部塞入视觉历史或循环记忆：演示大多为成功轨迹，把持久记忆放在动作策略内会令重试动作耦合到此前失败轨迹的潜状态；
  2. 将 VLA 接入带规划器、几何原语、深度或标定几何的工具箱：系统存在多条运动控制路径，效果增益究竟来自更强感知、替代运动原语还是 VLA 本身难以归因；仅靠纯文本子任务指令又容易掩盖策略的真实运动能力。
- **2AM 的受控设计点**：「更少的工具广度，更高的接口带宽（less tool breadth, but greater interface bandwidth）」。部署通路仅允许接收：任务语言 $g$、全局视角 RGB $I^a$、腕部 RGB $I^w$、机器人本体状态 $s_t$；**禁止**深度图、在线分割掩码、物体位姿、像素反投影 3D、或任何解析式物体运动规划。所有任务相关运动均须通过同一个分段无状态（episodically stateless）的 Action Model 执行。

### 2. 双模型分工与组合引导契约（§2.2–2.3, Eq. 1–3, Table 1）
- **Agent Model**（按短动作块时钟 $k$ 更新）：持有并整合历史记录 $H_k$（筛选过的观测、历史指令与可见结果），输出引导命令：
  $$c_k = \mathrm{AM}_{\mathrm{agent}}(g, I_k^a, H_k) = (\ell_k, q_k^{\mathrm{grasp}}, q_k^{\mathrm{place}}, q_k^{\mathrm{move}})$$
- **Action Model**（在动作块内高频闭环）：不接收 $H_k$，跨调用不保留回合状态，仅依据当前全局 RGB $I_t^a$、腕部 RGB $I_t^w$、状态 $s_t$ 与 $c_k$ 预测 $L=16$ 步绝对末端执行器动作：
  $$\hat{a}_{t:t+L-1} = \mathrm{AM}_{\mathrm{action}}(I_t^a, I_t^w, s_t, c_k)$$
- **4 个接口字段的尺度分工（Table 1）**：
  - $\ell_k$（Language）：规定当前语义阶段的行为（如 "place the bowl on the plate"），不负责实例消歧；
  - $q_k^{\mathrm{grasp}} \in [0,1000]^2$（`grasp_target`）：绑定下一个目标实例，仅在接触前有效，物体被抓起或阶段切换后应省略，避免把控制器拉回原位；
  - $q_k^{\mathrm{place}} \in [0,1000]^2$（`place_target`）：跨搬运阶段的长期目标位置约束，解决腕部视角看不到远端目标或被机械臂遮挡的问题；
  - $q_k^{\mathrm{move}} \in [0,1000]^2$（`move_target`）：近期方向引导（未来若干步的期望末端投影位置），仅作为方向提示而非必须严格经过的路径点。
- **缺省语义**：缺失字段直接省略而非使用特殊坐标点，因为各提示的可用性本身依赖当前操作阶段。

### 3. 无模型教师的离线监督与抗噪训练（§3.1–3.3, Eq. 4–7）
- 不依赖 VLM 自动旁白标注，而是从同步示教（RGB、机器人状态、阶段标签、物体框/掩码、夹爪接触与物体运动）确定性恢复监督信号（特权标注仅用于离线构造训练标签，部署期不可见）：
  - `grasp_target` / `place_target` 取可见物体包围盒归一化中心：$q = 1000\left(\frac{x+w/2}{W}, \frac{y+h/2}{H}\right)$（Eq. 4）；
  - `move_target` 从末端执行器投影轨迹 `robot_xy` 采样未来窗口：$q_t^{\mathrm{move}} = \texttt{robot\_xy}_{t+\Delta}$，$\Delta \sim \mathcal{U}\{10,\dots,20\}$（Eq. 5），以适应不固定的 Agent 推理延迟；
  - 动作块边界允许跨越子任务标签边界，避免策略在抓取/抬起/释放衔接处停顿。
- **应对不完美引导的三类训练期扰动（§3.2, Eq. 6–7）**：
  1. **非空条件随机掩码（Structured hint dropout）**：$m_t \sim p_{\mathrm{mask}}(m \mid v_t)$，满足 $m_t \preceq v_t$ 且 $\|m_t\|_0 \ge 1$，使纯语言、纯点位及各类混合引导都能训练，防止模型把多种提示绑定为不可拆分的整体；
  2. **各向同性高斯坐标噪声**：$\widetilde{q} = \operatorname{clip}_{[0,1000]^2}(q + \epsilon)$，$\epsilon \sim \mathcal{N}(0, \sigma^2 I_2)$；
  3. **未来窗口采样**应对时间抖动。
- Action Model 实现采用 `Qwen3-VL-4B` 主干 + 流匹配（flow-matching）Action Expert，动作块长度 $L=16$。

### 4. LIBERO-Mem 主实验与消融（§4.1–4.2, Table 2–3）
- 评测基准为 LIBERO-Mem 全部 10 项任务（涵盖运动 M、序列重复 S、关系 R、遮挡 O 四类），指标包括 Strict SR（严格按顺序与次数停止，超次即失败）、Relaxed SR（曾在超调前完整达到目标序列）与 Completion（有序子目标完成比例）。
- **核心汇总对比（Table 3）**：
  - 文献已发表最佳基线（SlotSSM）：Strict SR `0.00%`、Completion `14.80%`（文献版 $\pi_0$ 与 SlotVLA 仅在 T1 达 `50.0%`、其余 0，平均 `5.0%`）；
  - 作者对齐复现的 $\pi_0$（同用 `Qwen3-VL-4B` 主干、双 RGB 与 16 步动作块）：Strict SR `12.25%`、Relaxed SR `37.42%`、Completion `70.79%`；
  - 纯语言接口消融（`Language-only, zero-shot (Qwen3.8-27B)`）：Strict SR `7.25%`、Relaxed SR `19.42%`、Completion `53.72%`；
  - **2AM 原版零样本系统**（`Language + hints, zero-shot (Qwen3.8-27B)`）：Strict SR `11.83%`、Relaxed SR `63.00%`、Completion `76.29%`（相对纯语言消融：Strict +4.58 pp、Relaxed +43.58 pp、Completion +22.57 pp；相对对齐 $\pi_0$：Completion +5.50 pp、Relaxed +25.58 pp，Strict 略低 0.42 pp）；
  - 后续 SFT 轻量 Agent（`Language + hints, SFT (Qwen3-VL-4B)`）：Strict SR `31.90%`、Relaxed SR `62.20%`、Completion `80.44%`。
- **关键任务分布现象（Table 2）**：2AM 在 10 个任务上 Relaxed SR 全部高于复现 $\pi_0$，在 7 个任务上 Completion 更高；在关系/遮挡密集型任务 T7–T10 上优势明显（如 T9 `47.50%` vs `0.00%` Strict/Relaxed SR），但在重复计数任务 T1–T6 上 Strict SR 低于 $\pi_0$ 复现（如 T1 `1.67%` vs `30.83%`），暴露出「推进可达，精确语义终止仍难」的问题（Relaxed `63.00%` vs Strict `11.83%`）。

### 5. 项目页（2026-11 AgentMem 修订版）补充事实与开源演进
- 项目主页于 2026-11 将系统演进命名为 **AgentMem**，Agent 记忆进一步显式分解为三类组件：历史关键帧（Visual Evidence）、已下发子目标历史（Decision History）与紧凑语义事实（Persistent Semantics）。
- 扩展至 4 个仿真基准共 42 项任务（均采用统一训练配方、分基准训模型）：RoboMME `70.13%` SR（16 tasks）、RMBench `62.33%` SR（9-task common subset）、LIBERO-Mem `77.07%` Completion（10 tasks）、RoboDojo-Mem `30.33%` SR（6 tasks）。
- 在 6 项仿真任务（RoboMME 与 RMBench 子集，3 种子 × 每任务 50 次）上进行记忆与引导正交消融：无持久记忆+纯语言 `17.78 ± 1.58%`；无持久记忆+引导 `23.78 ± 1.84%`；完整记忆+纯语言 `22.89 ± 0.77%`；**完整记忆+引导 `67.00 ± 1.53%`**（仅语义事实记忆而无历史图像为 `56.89%`）。
- 真机实验对比（$\pi_{0.5}$ / $\pi_{0.5}$·8 frames / AgentMem）：Cup Reveal `5/15` / `5/15` / `9/15`；Block Swap `0/15` / `15/15` / `15/15`；Pattern Lock `0/16` / `0/16` / `15/16`。

### 6. 局限性（§6）
- arXiv v2 论文实验仅在单个仿真基准 LIBERO-Mem 上验证；
- 2D 图像引导无法表达隐藏三维几何、接触力或连续运动历史；
- 对齐复现 $\pi_0$ 已达 70.79% 完成度，说明完成度指标本身不足以证明记忆问题已被解决，精确语义停机（strict termination）仍是独立瓶颈；
- 论文尚未独立消融单个引导字段（grasp / place / move）贡献、不同强弱 Agent 底座差异，以及引导更新频率在能力与延迟间的权衡。
