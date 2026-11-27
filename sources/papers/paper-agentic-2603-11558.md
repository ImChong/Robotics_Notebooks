# RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks

- **作者：** Ruiying Li, Yunlang Zhou, Yuyao Zhu, Kylin Chen, Jingyuan Wang, Sukai Wang, Kongtao Hu, Minhui Yu, Bowen Jiang, Zhan Su, Jiayao Ma, Xin He, Yongjian Shen, Yang Yang, Guanghui Ren, Maoqing Yao, Wenhao Wang, Yao Mu
- **机构：** 智元机器人（AgiBot）、上海交通大学（SJTU，含 MoE Key Lab of Artificial Intelligence, AI Institute）、新加坡国立大学（NUS）；官方仓库合作机构标识另含 ScaleLab
- **版本 / 载体：** arXiv:2603.11558v3 [cs.RO]（2026-03-12 提交 v1，2026-04-01 更新 v3）
- **原文：** https://arxiv.org/abs/2603.11558（HTML 全文：https://arxiv.org/html/2603.11558）
- **项目页 / 代码：** 项目主页 https://roboclaw-agibot.github.io/ ；官方 GitHub 仓库 https://github.com/RoboClaw-Robotics/RoboClaw（经 Step 2.5 核查判定为**部分开源**：核心 Agent 编排层、Skill 规范、MCP 服务示例与 TUI/GUI 入口已公开；README 注明完整 VLA 模型部署所需的 `.whl` 包需私信申请，当前主要在 Agibot G01 平台验证）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§1 Introduction、§2 Related Works、§3.1–3.3 Method、公式 (1)–(8)、Table 1–3、Fig. 1–5 题注、§4.1–4.4 Experiments、§5 Conclusion、References），并核查官方 GitHub 仓库 README 与文件目录索引
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [RoboClaw 实体页](../../wiki/entities/paper-agentic-2603-11558.md)

### 1. 问题定义与整体架构（§1, §3.1）
- 传统 VLA 流程将数据采集、策略训练与部署执行拆分为独立阶段，导致大量人工重置环境、不同人员/阶段间的任务语义与状态边界不一致，以及训练状态分布与部署执行条件脱节。
- **RoboClaw** 使用现成 VLM 作为高层元控制器（meta-controller），在上下文学习（ICL）与思维链（CoT）推理范式下维持单一闭环，覆盖采集、学习与部署全生命周期。
- 系统按 OpenClaw 风格组织为三层抽象：
  1. **Skills（技能层）**：可复用的高层流程（如 `eap-data-collection`、`long-horizon-execution`、`monitored-subtask-execution`），负责编排工具。
  2. **Tools（工具层）**：通过 Model Context Protocol（MCP）暴露的标准化接口（如 `Start Policy`、`Terminate Policy`、`Change Policy`、`Env Summary`、`Fetch Robot Stats`、`Call Human`）。
  3. **Policies（策略层）**：底层 VLA 基础模型（本文采用 $\pi_{0.5}$），输出关节动作块。
- 结构化记忆状态 $m_t = (r_t, g_t, w_t)$（Eq. 1）：
  - $r_t$：角色身份（当前工作模式与可用工具集）；
  - $g_t$：任务级记忆（全局任务、拆解后的子任务及完成状态）；
  - $w_t$：工作记忆（短期上下文、当前激活技能与工具调用历史）。

### 2. 纠缠动作对（EAP）与底层策略（§3.2, Table 1）
- 子任务选择（Eq. 2）：$z_t = \operatorname{RoboClaw}(m_t, o_t) \in \mathcal{Z}$。
- 底层策略（Eq. 3–5）：$A_t = \pi_{0.5}(o_t, l_t, q_t) = [a_t, \dots, a_{t+H-1}]$，其中指令 $l_t$ 由 RoboClaw 在 MCP 调用时动态生成而非人类直接下发；采用条件流匹配（conditional flow matching）目标 $\mathcal{L}^\tau(\theta) = \mathbb{E}[\|v_\theta(A_t^\tau, o_t, l_t, q_t) - u(A_t^\tau \mid A_t)\|^2]$ 训练，$A_t^\tau = (1-\tau)\epsilon + \tau A_t$。
- **Entangled Action Pairs（EAP，Eq. 6–8）**：为每个技能 $k$ 学习一对正向策略 $\pi_{\theta_k}^\rightarrow$ 与逆向重置策略 $\pi_{\phi_k}^\leftarrow$，交替采集 $\tau_k = (\tau_k^\rightarrow, \tau_k^\leftarrow)$ 并存入数据集 $\mathcal{D}$。逆向任务刻意设计得比正向任务更简单，以保证自重置循环稳定性。
- **$\pi_{0.5}$ 微调超参数（Table 1）**：`bfloat16` 精度、batch size `16`、训练 `10k` 步、warmup `100` 步、学习率 `2.5e-5`、推理步数 `3`、开启 gradient checkpointing；LoRA 配置为 rank `r=16`、alpha `16`、dropout `0.1`、target modules `all-linear`。

### 3. 部署监督与失败学习（§3.3, §4.4）
- 部署阶段复用同一代理循环，通过 `Fetch Robot Stats` 与 `Env Summary` 周期性评估进展、写入 $w_t$ 并更新 $g_t$。
- 将失败分为两类（§4.4）：
  - **非退化失败（Non-degrading failures）**：环境状态未显著改变（如空抓但药水瓶仍直立），直接重试当前策略即可。
  - **退化失败（Degrading failures）**：环境状态已被破坏并偏离前置条件区域（如药水瓶倾倒或滑出工作区），需通过 `Change Policy` 触发恢复策略或重规划；若无法自主恢复或触发安全约束，则通过 `Call Human` 请求人工介入。
- 部署轨迹与人工接管后的恢复轨迹均回流至 $\mathcal{D}$，逐步沉淀为专用恢复策略并持续迭代正向策略。

### 4. 真机实验结果（§4.1–4.3, Table 2–3, Fig. 4–5）
- **平台**：Agibot G01 双臂移动操作机器人（不含末端共 20 自由度，每臂配 1 自由度 AGIBOT OmniPicker 自适应夹爪）。
- **场景与单技能任务**：4 个真实整理/取物场景（卧室梳妆台、厨房货架、书桌、便利店货架）；4 个单技能评测任务为 Body Lotion 放置、Primer 放抽屉并关抽屉、Lipstick 窄槽插接、Tissue Wipe 擦拭水渍。
- **采集效率（§4.1, Fig. 4a–b）**：采集等量数据时，人工基线所需人力时间约为 RoboClaw 的 `2.16×`（全生命周期人力时间投入减少 `53.7%`）；rollout 阶段人工基线所需干预次数约为 RoboClaw 的 `8.04×`。
- **逆向重置策略成功率（Table 2，每项 50 次）**：Body Lotion `36/50`、Primer `38/50`、Lipstick `43/50`、Tissue Wipe `39/50`。
- **正向策略跨轮次迭代成功率（Table 3，Iter 1–5，每轮增加 50 条样本，每项 50 次测试）**：
  - Body Lotion：`21/50` → `25/50` → `32/50` → `37/50` → `43/50`
  - Primer：`23/50` → `31/50` → `31/50` → `34/50` → `40/50`
  - Lipstick：`2/50` → `4/50` → `11/50` → `16/50` → `23/50`
  - Tissue Wipe：`11/50` → `13/50` → `14/50` → `21/50` → `26/50`
- **长程任务（§4.3, Fig. 4c）**：梳妆台整理（组合 4 个子任务，20 次试验均值）相比无 RoboClaw 框架的同数据集 $\pi_{0.5}$ 基线与四子任务成功率乘积基线，长程任务成功率提升 `25%`。

### 5. 局限性（§5）
- 云端大 VLM 控制器可能引入推理延迟；
- EAP 依赖能够构造可复现前置状态空间的实用逆向重置动作假设；
- 开源仓目前主要围绕 Agibot G01 验证，完整 VLA 部署 `.whl` 尚未直接随公共代码树分发。
