# DeepInsight: A Unified Evaluation Infrastructure Across the Physical AI Stack（arXiv:2606.17574）

> 来源归档（ingest）

- **标题：** DeepInsight: A Unified Evaluation Infrastructure Across the Physical AI Stack
- **类型：** paper / Physical AI / 评测基础设施 / 人形全栈
- **机构：** XPENG Robotics（小鹏机器人）
- **arXiv：** <https://arxiv.org/abs/2606.17574>（PDF：<https://arxiv.org/pdf/2606.17574>）
- **通讯作者：** Jie Chen（chenj81@xiaopeng.com）
- **入库日期：** 2026-06-18
- **一句话说明：** XPENG Robotics 提出的 **Physical AI 全栈统一评测基础设施**：在 **单一 runtime** 上贯通 **System 2（基础模型推理）→ System 1（视运动策略）→ System 0（全身控制）**，用 **task / resource / result** 三抽象与 **统一 trace** 实现跨层回归定位，而非各层拼接独立 harness。

## 摘要级要点

- **问题：** Physical AI 评测算子跨度超过 **三个数量级**（单次 decode 到数千 physics tick）；episode 长度、模态、奖励语义、资源画像正交变化；现有框架（lm-eval、OpenCompass、Inspect AI、VLMEvalKit、Isaac Lab 等）各自只覆盖谱段一端。
- **统一动机：** 部署栈中 **层间失败耦合**——语义规划错误改变策略分布、策略犹豫改变稳定器工况、稳定器恢复改变高层可尝试动作；分段 harness 保留局部 benchmark 有效性，但丢失 **共享 run identity、资源记账与 trace 连续性**，无法诊断跨层回归。
- **三不变量：** **一个 episode driver**、**一个 resource-handle 协议**（LLM 推理与 sandbox/仿真后端均实现）、**一个 trace identity scheme**（所有子系统写入同一结构化记录）。
- **生产部署：** 已在 XPENG 人形 embodied stack **三层生产环境**运行；新 benchmark 多数以 **配置** 接入（suite + template 快照合并持久化）。
- **System 2 量化：** 对 lm-eval / Inspect AI / VLMEvalKit / lmms-eval 在 Qwen3.6-27B 等模型上 **复现 reference 读数**；单节点 **1.03×–1.29×** 墙钟加速；**1→4 节点近线性**（27-suite 工作负载 **4.00×**）。
- **全栈案例：** System 1 闭环仿真 + 主观偏好评测；System 0 **trace 驱动发布决策**（聚合筛选后行为级诊断可否决 aggregate winner）；**System 2–1–0 组合任务**（展厅 vehicle-guide）在 **96 episodes** 上 **60.4%** 端到端成功率，**38** 失败 episode 中 **42.1%** 归因 System 1、**28.9%** boundary handoff。

## 核心论文摘录（MVP）

### 1) Physical AI 评测谱与 System 2/1/0 分层

- **链接：** <https://arxiv.org/abs/2606.17574> §1, Figure 1
- **摘录要点：** 论文将 embodied humanoid stack 分为三层：**System 2** 语义目标推理（FM 评测：MMLU、SWE-bench、GAIA 等）；**System 1** 视运动策略执行（导航/操作：CALVIN、LIBERO、SimplerEnv 等）；**System 0** 全身稳定与控制（HumanoidBench、Isaac Lab 等）。算子正交轴：episode 长度、观测模态、奖励语义、资源画像（GPU 推理 / I/O sandbox / 物理并行仿真）。
- **对 wiki 的映射：**
  - [DeepInsight](../../wiki/entities/deepinsight.md) — 实体总览与三抽象架构。
  - [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md) — 闭环评测与跨层诊断语境。

### 2) 三抽象：task / resource / result

- **链接：** §3, Figure 2–4
- **摘录要点：**
  - **Task：** 无状态 environment + **per-episode handle**（`reset/step` 两方法）；judgment 与 task **正交**（四类 scorer：rule / model / vision / benchmark-specific）；配置快照（suite + template）为持久化 artifact。
  - **Resource：** inference plane（vLLM / vLLM-Omni / SGLang 异构引擎池）与 sandbox plane（预热的无状态池 + per-episode OpenSandbox 容器；physics simulator 亦走 sandbox plane）对称 **acquire/release** 协议；各 stage **独立并发预算**。
  - **Result：** 统一 **TraceRecord** schema（`run_id, suite_id, task_id, episode_id, epoch` + multi-rate `turn_id/step_id/tick` + `event_id/parent_event_id`）；trace 为保留单元，reporting 为下游 reader。
- **对 wiki 的映射：**
  - [DeepInsight](../../wiki/entities/deepinsight.md) — 架构表与 Mermaid 流程图。

### 3) System 2 对标与吞吐机制

- **链接：** §4, Tables 2–5, Figures 5–8
- **摘录要点：** 在 8×A100 单节点、共享 vLLM 与 judge 配置下，DeepInsight 在 text / multimodal VQA / omni-modal 对齐表上 **closest-to-Ref 行数最多**；相对 Inspect AI **1.13×**、lm-eval **1.29×** 墙钟加速。内部消融：**async pipelining** 在 AIME-2024 上 **1.41×**；**stage decoupling** 在 LiveCodeBench v6 上 **3.31×**；多节点 **~2× per doubling**。
- **对 wiki 的映射：**
  - [DeepInsight](../../wiki/entities/deepinsight.md) — System 2 对标摘要表。

### 4) System 1/0 与全栈跨层诊断

- **链接：** §5, Tables 6–8, Figures 10–13
- **摘录要点：**
  - **System 1：** VLN-CE / LIBERO 风格闭环 + 音频条件动作生成 **主观 pairwise 偏好**（20 clips × 10 raters）写入同一 result schema。
  - **System 0：** 两阶段 **发布工作流**——SR–MPJPE 平面筛选 → 轨迹统计行为诊断；aggregate winner **WBC-RC-01** 在 hip dynamics / contact attitude / upper-body ROM 上 **Fail**。
  - **全系统：** vehicle-guide 组合任务；子目标边际成功率可达 **90%+** 但端到端仅 **60.4%**；主要失败 **System 1 navigation (42.1%)** 与 **boundary handoff (28.9%)**。
- **对 wiki 的映射：**
  - [DeepInsight](../../wiki/entities/deepinsight.md) — 案例研究与跨层归因表。
  - [loco-manipulation](../../wiki/tasks/loco-manipulation.md) — 同仓库内 System 2/1/0 叙事对照。

## 对 wiki 的映射

- 主实体页：[`wiki/entities/deepinsight.md`](../../wiki/entities/deepinsight.md)
- 概念互链：[仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)、[数据飞轮](../../wiki/concepts/data-flywheel.md)
- 栈地图：[训练栈分层技术地图](../../wiki/overview/robot-training-stack-layers-technology-map.md)
- 仿真参照：[Isaac Lab](../../wiki/entities/isaac-lab.md)、[Genesis World 1.0](../../wiki/entities/genesis-world-10.md)
- 方法参照：[VLA](../../wiki/methods/vla.md)
