# Robo-Cortex: A Self-Evolving Embodied Agent via Dual-Grain Cognitive Memory and Autonomous Knowledge Induction

- **作者：** Nga Teng Chan, Yi Zhang, Yechi Liu, Renwen Cui, Fanhu Zeng, Zeyuan Ding, Xiancong Ren, Zhang Zhang, Qifeng Chen, Jian Liu（通讯作者）, Yong Dai, Xiaozhu Ju（项目负责人）
- **机构：** 香港科技大学（HKUST）、中国科学院自动化研究所（CASIA）、北京人形机器人创新中心（X-Humanoid）、北京航空航天大学（BUAA）
- **版本 / 载体：** arXiv:2605.18729v1 [cs.RO]（2026-05-18 提交）
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2605.18729)、[arXiv 摘要页](https://arxiv.org/abs/2605.18729)
- **项目页 / 代码：** 未确认开源代码（原文首页列出项目页 `https://robocortex66.github.io`，但未提供 GitHub 源码仓库链接；截至 2026-11-27 抓取项目页状态异常 `status=0` 且无候选代码仓库）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§1 Introduction、§2 Related Work、§3.1–§3.4 方法与公式 (1)–(10)、§4.1–§4.4 实验设置与 Table 1–3 / Fig. 1–5 图注及正文分析、§5 Conclusion and Limitations、References）；补充材料未随 arXiv HTML 附带
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[Robo-Cortex 实体页](../../wiki/entities/paper-agentic-2605-18729.md)

### 1. 研究动机与核心问题
- 针对具身导航中的 **经验失忆（experiential amnesia）**：现有基于轨迹回放或局部反应式策略的方法虽能在回合内利用记忆或推理，但难以将历史交互抽象为可跨场景迁移的通用决策策略。
- 提出 **Robo-Cortex** 自演化框架，包含两条评测范式：静态版 **Robo-Cortex**（评测前固定记忆与启发式库）与在线自适应版 **Robo-Cortex++**（推理阶段持续在线更新启发式库）。

### 2. 三大核心模块
1. **Imagine-then-Verify 闭环规划（§3.2）**：
   - 每步由规划器生成 $N$ 个候选方案 $p_t^{(i)} = (a_t^{(i)}, r_t^{(i)})$，子任务分配器 $\mathcal{D}(\cdot)$ 将低层离散动作序列 $a_t^{(i)}$ 与高层语义推理 $r_t^{(i)}$ 对齐为 $\mathcal{S}_t^{(i)}$。
   - 世界模型 $\mathcal{W}(\cdot)$（`Wan2.1-I2V-A14B-480P-Diffusers`）预测短视界未来视觉推演 $\hat{o}_{t+1:t+h}^{(i)}$，VLM 评估器（`Qwen2.5-VL-72B-Instruct-AWQ`）按目标 $g$ 打分选出 $p_t^*$ 并高频重规划，避免长程开环推演误差累积。
2. **双粒度认知记忆（Dual-Grain Cognitive Memory，§3.3）**：
   - 统一组织为记忆图 $\mathcal{G}=(\mathcal{V},\mathcal{E})$，节点分为回合根节点 $\mathcal{V}^{\text{root}}$、决策步节点 $\mathcal{V}^{\text{traj}}$ 与子任务语义单元 $\mathcal{V}^{\text{sub}}$，包含层级父子边与时序边。
   - **短期反思记忆（SRM）**：对最近 $w$ 个子任务滑动窗口 $W_t$ 总结局部进展、重复失败模式与子目标上下文 $s_t^{\text{SRM}}$，注入同回合后续规划。
   - **长期原则记忆（LPM）**：回合结束后沿时序边前向收集下游轨迹 $\tau_i$（长度 $k$），分别提炼成功回合的指导原则（guiding principles）与失败回合的警示原则（cautionary principles）；测试时在目标条件状态嵌入空间中按阈值 $\delta_{\text{lpm}}$ 做最近邻检索。按目标聚合记忆时优先保留成功且轨迹更短的回合；无成功记录时保留多样化失败轨迹。
3. **自主知识归纳（Autonomous Knowledge Induction, AKI，§3.4）**：
   - 回合结束后由 VLM 启发式提取器从记忆图提取结构化启发式 $h = (\rho, d, a, c, y)$（模式标识 $\rho$、问题描述 $d$、推荐策略 $a$、置信度 $c \in [0,1]$、成败标签 $y \in \{\text{success}, \text{failure}\}$）。
   - 按 $\rho$ 与语义相似度聚类后由合并模型生成泛化启发式 $\tilde{h}_{\mathcal{C}} = (\rho, \tilde{d}, \tilde{a}, \tilde{c})$ 写入 Navigation Heuristic Library，并将高置信度规则回注到 SRM 与规划器中形成反思-适应闭环。

### 3. 实验关键数据核对（§4）
- **仿真与底座**：Habitat-sim；IGNav 与 AEQA 基于 HM3D 验证集，AR 基于 MP3D；动作空间离散化为前进 $0.20\text{ m}$、左右旋转 $22.5^\circ$；VLM 为 `Qwen2.5-VL-72B-Instruct-AWQ`（多 vLLM 服务共享权重），世界模型为 `Wan2.1-I2V-A14B-480P-Diffusers`。
- **Table 1（主实验对比）**：
  - `World-In-World`：IGNav SR `38.57` / SPL `27.50` / Mean Traj `47.10`；AR SR `20.68` / Mean Traj `7.09`；AEQA Score `27.19` / Mean Traj `20.53` / SPL `20.56`。
  - `Robo-Cortex`（静态）：IGNav SR `41.26` / SPL `31.66`（较前序最优基线 `+4.16%` SPL）/ Mean Traj `49.90`；AR SR `22.39` / Mean Traj `6.97`；AEQA Score `29.78` / Mean Traj `17.34` / SPL `21.40`。
  - `Robo-Cortex++`（在线更新）：IGNav SR `45.07` / SPL `35.06` / Mean Traj `47.50`；AR SR `23.88` / Mean Traj `7.57`；AEQA Score `30.59` / Mean Traj `16.61` / SPL `25.57`。
- **Table 2（Seen split 记忆累积）**：Basic Pipeline 在 Round 1 仅 `36.11` IGNav SR；加入 SRM 升至 `40.18`；Round 3 加入 LPM & SRM 达 `44.29` IGNav SR、`24.03` AR SR、`31.71` AEQA Score（SPL `27.51`）。
- **Table 3（Unseen split 启发式迁移，不启用 LPM）**：Basic Pipeline 在 IGNav 上为 `34.72` SR / `24.03` SPL；迁移 seen split 启发式且固定（`w/ transferred heur.`）达 `48.61` SR / `39.33` SPL（`+15.30%` SPL），而继续在线更新（`w/ transferred heur. + update`）为 `41.67` SR / `34.84` SPL，说明在少量未见样本上激进在线改写可能覆盖高质量先验。
- **真机初步实验（§4.4）**：客厅与超市两个室内 IGNav 场景中，Basic Pipeline 成功率为 `4/10`；典型失败为视觉线索失配后继续盲目执行，引入 SRM 与仿真归纳启发式后可检测失配并回退至最近目标一致区域恢复任务（Fig. 5）。
