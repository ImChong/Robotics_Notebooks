# MEMORA: Embodied Action Memory from Egocentric Videos for Reasoning and Planning

- **作者：** Zihao Yu, Xiu Yuan, Chongjie Zhang
- **机构：** Washington University in St. Louis
- **版本 / 载体：** arXiv:2607.14252v2 [cs.RO]（2026-08-31 修订，v1 提交于 2026-07-15）；v1 获 RSS 2026 Workshop on Foundation Models for Robot Planning (FM4RoboPlan) Oral；v2 已被 EMNLP 2026 录用
- **原文：** https://arxiv.org/abs/2607.14252（全文核对使用 arXiv HTML 版 https://arxiv.org/html/2607.14252）
- **项目页 / 代码：** https://github.com/yuzihaowashu/MEMORA | https://yuzihaowashu.github.io/MEMORA/（MIT License，已开源：包含记忆构建管线 `src/memora/pipeline/`、多存储检索与 ReAct 代理 `src/memora/memory_agent/`、EAM-QA 与 MEMORA-Planning 评测入口 `src/memora/evaluation/`、基准任务 `src/memora_bench/` 与结果复现脚本 `scripts/paper_results/`；18 名参与者四存储记忆与已保存模型输出托管于 HuggingFace `DependableDavid/MEMORA`，派生基准数据继承 EPIC-KITCHENS CC BY-NC 4.0 协议，原始第一人称视频需从 EPIC-KITCHENS 获取）
- **核对范围：** 通读 arXiv:2607.14252v2 全文（Abstract、Sections 1–7、Limitations、Ethical Considerations、References、Appendices A–F、Tables 1–24、Figures 1–20 题注与示例框）并静态核对官方仓库 https://github.com/yuzihaowashu/MEMORA 的 `README.md` 与 `files_sample` 目录结构。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[MEMORA 实体页](../../wiki/entities/paper-agentic-2607-14252.md)

1. **问题形式化：具身行动记忆（Embodied Action Memory, EAM）**
   - 作者将 EAM 形式化为经验形成与决策使用分离的双映射生命周期：$\mathcal{M}_t = F(\mathcal{M}_{t-1}, o_t)$ 与 $d_\tau = G(\mathcal{M}_\tau, x_\tau)$，其中 $o_t$ 为时间 $t$ 的具身观测，$\mathcal{M}_t$ 为持续演化的记忆状态，$x_\tau$ 为后续推理问题、规划目标或决策上下文。
   - 与传统长视频问答（推理时直接输入原始视频）或纯文本对话记忆不同，EAM 要求智能体从跨会话的连续第一人称视频流中在线建构并修订持久化记忆，在测试时不回看原始视频仅凭记忆支撑溯因问答与前瞻规划。

2. **四类型世界记忆与“在线编辑 + 离线巩固”双时间尺度机制（Section 4 & Appendix A）**
   - **分段编码（Perception Processor / Segment Encoder $\phi$）**：将视频切分为互不重叠的 10 秒片段 $v_t$（按 2 FPS 解码、上限 24 帧，联合原生音频流输入 Qwen2.5-Omni-7B），结合前一轮文本工作记忆摘要 $c_{t-1}$，输出结构化观测 $z_t = (z_t^{\text{env}}, z_t^{\text{ent}}, z_t^{\text{act}})$。
   - **四类型存储（Typed Stores）**：
     - **Environment Memory ($\mathcal{M}_t^{\text{env}}$)**：按稳定 `location_id` 确定性合并，记录区域布局、命名分区与空间关系（如 `drying_rack RIGHT_OF sink_basin`）及更新历史。
     - **Entity Memory ($\mathcal{M}_t^{\text{ent}}$)**：由在线记忆编辑器（Online Memory Processor，主配置为 Qwen3-30B-A3B-Instruct-2507）执行主动编辑 $\delta_j \in \{\textsc{Add}, \textsc{Upd}, \textsc{Del}, \textsc{Noop}\}$，维护物体视觉属性、空间位置、当前状态与完整 `state_history`；实体库超长时以密集检索预筛候选实体。
     - **Activity Memory ($\mathcal{M}_t^{\text{act}}$)**：按 10 秒片段追加（append-only），保存动作摘要、叙事与细粒度 `action_breakdown`。
     - **Inferred Knowledge ($\mathcal{M}^{\text{inf}}$)**：分两步离线生成——Offline Consolidation Processor（复用 Qwen3-30B-A3B-Instruct-2507）跨视频总结存储偏好、组织习惯与工作流模式；Inferred Knowledge Enrichment Processor（Qwen3.6-35B-A3B + 规则骨架）进一步提炼 `routine_skills`、`generated_preferences`、检索文本与支撑证据 ID。
   - **压缩与编辑行为统计（Figure 2, Appendix A.3, Table 21）**：在 18 名参与者上，在线编辑相对无编辑追加策略实现 **中位数 18.1×**（正文中约 18×）的实体记录压缩（最小 9×，P12/P23/P25 超过 30×，其中 P23 达 44.8×）。P01–P04 校准集（$n=37{,}648$ 次物体观测）中编辑器决策分布为：`Noop` 45.8%（17,244）、`Update` 44.4%（16,718）、`Add` 8.0%（2,995）、`Delete` 1.8%（691），表明超过 90% 的操作用于维护既有实体或修正误检而非盲目扩张条目。

3. **类型感知检索与规划执行（Section 4.2, Appendix C.2, Appendix F）**
   - 采用基于 `intfloat/e5-base-v2` 的密集检索（top-$k=10$，余弦相似度 $\ge 0.95$ 去重）与 ReAct 迭代循环（EAM-QA 最多 5 轮，Planning 最多 8 轮含强制输出最终 `Plan:` 轮）。
   - Planning 中全量 MEMORA 优先调用 `get_routine_skill` 与 `get_preferences` 获取跨事件程序性骨架与偏好，再以 `search_objects` / `get_object_history` 绑定参与者专属实体和地点（“先方法骨架、后物理接地”）。全流程使用 vLLM 0.19.1 部署开源权重模型；在 Qwen3.6-35B-A3B 上单目标端到端规划平均耗时 10.40 s（其中 LLM 生成 5.74 s、记忆工具调用 1.06 s，Table 24）。

4. **MEMORA-Bench 与核心实验结果（Section 5, Appendices B, C, E）**
   - **基准构成**：基于 EPIC-KITCHENS-100 扩展集 18 名参与者（201 段视频，约 45 小时），包含：
     - **EAM-QA**：2,763 道五选一题目（2,212 可答 + 551 跨参与者不可答 E-correct 对照；选项 E 为“信息不可用”），涵盖 `SPref`（偏好）、`SHabit`（触发-后续习惯）、`SRoutine`（多步策略）与 `ERecall`（单视频事件回忆）及 2/3-hop 推理变体。
     - **MEMORA-Planning**：207 个 `Replay` 任务（5 类厨房操作）与 153 个 `Generalize` 任务（`transfer` / `composition` / `fully-novel` 三类分布外设计），由规则化指标 `OrderExec`、`KeyObj`、`PrefAdh` 的无权均值构成 **Robot-Grounded Plan score (RGP)**。
   - **EAM-QA 结果（Table 3, 9, 10）**：在 `parametric-missed + MEMORA-committed` 诊断视角下，Gemma-4-31B-it 上 MEMORA 总体准确率达 **74.5%**（SPref 76.8%、SHabit 60.7%、SRoutine 81.5%、ERecall 79.8%），领先最强非 MEMORA 基线 `Flat-1D (processed)`（54.0%）达 **+20.5 个百分点**，相比无离线巩固的 `MEMORA-Episodic`（60.1%）提升 **+14.4 个百分点**（其中 `SRoutine` 提升 +24.6）。四款开源底座（Gemma-4-31B-it、Qwen3.6-27B 69.1%、Gemma-4-26B-A4B-it 54.1%、Qwen3.6-35B-A3B 12-PID 63.5%）上巩固增益均稳定为正。同时 Table 9 如实报告全集准确率（如 Gemma-4-31B 全集 46.1%，与非 MEMORA 最佳基线差 -0.5），反映未作答拒答率（E-rate）与提交后准确率的权衡。
   - **Planning 结果（Figure 3, Table 16, Figure 15）**：MEMORA 系列在 4 个底座 × 2 个评测集的全部 8 个组合中均位列第一；在 `Generalize` 上相对最强非 MEMORA 基线提升达 **+16.4%（Gemma-4-31B-it）至 +16.6%** 相对 RGP（在 Gemma-4-31B-it 上 `OrderExec` +43%、`PrefAdh` +14%），且在 `Generalize` 上所有非 MEMORA 检索基线（Flat-1D / Graph-2D）均低于无记忆 `Parametric` 基线，印证原始块检索会引入上下文污染。在较强推理模型（Qwen3.6-35B-A3B +0.025、Gemma-4-31B-it +0.022）上，离线巩固相对 `MEMORA-Episodic` 带来明显增益。
   - **受控消融（14-PID 留出集 2,017 题，Tables 20–22）**：将在线编辑模型由 Qwen3-30B-A3B 降为 Qwen3-14B（`A1-14B`）使 Gemma-4-26B 准确率下降 **5.8 个百分点**、E 选率由 42.8% 升至 62.3%（表 21 显示写入状态历史事件减少 24.5%）；将音视频感知 Qwen2.5-Omni-7B 换为纯视觉 Qwen2.5-VL-7B（`A2-VL7`）仅下降 **2.7 个百分点**，说明在线记忆编辑器容量是比音频感知更关键的瓶颈。将离线丰富模型替换为 `gpt-4o-mini` 仅产生 -0.16 至 +0.6 百分点波动，证明纯开源管线无明显精度损失。
   - **真机演示（Section 5.4, Appendix D）**：在 Unitree G1 桌面场景中仅以人类第一人称演示视频形成 MEMORA 记忆（无机器人示教），通过固定规则底层控制器执行语言规划，在 `Prepare Drink`（选可口可乐瓶与橙色塑料杯）与 `Breakfast`（按序取黄色碗、肉桂茶、黄色勺与面包）上实现 **2/2 一次成功**。

5. **主要局限（Limitations & Appendices A.7, C.6, D.2）**
   - 仅在 EPIC-KITCHENS-100 厨房场景（每名参与者最多 15 个会话）中验证，尚未覆盖数月以上长期漂移、跨领域迁移与主动遗忘策略。
   - 对前端感知和写入时编辑质量敏感，漏检物体或持续的感知误差可能固化进长期记忆；且基准评估的是已完成记忆库，未做单视频内前缀时间回滚测试。
   - 规则指标基于文本匹配，无法完全替代真机或高保真仿真中的闭环执行验证；Unitree G1 演示仅为 2 任务的定性接地检验，而非统计意义上的策略泛化基准。
