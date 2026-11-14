# X-Planner（arXiv:2609.25187）

> 来源归档（paper）

- **标题：** X-Planner: Event-Structured Task Planning for Embodied Intelligence
- **作者：** X Square Robot Team
- **机构：** 自变量机器人（X Square Robot）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.25187>
- **PDF：** <https://arxiv.org/pdf/2609.25187>
- **项目页：** <https://x-square-robot.github.io/Xplanner/>
- **代码：** <https://github.com/X-Square-Robot/Xplanner>
- **权重 / 基准：** <https://huggingface.co/x-square-robot/X-Planner-9B-0916>（Apache-2.0）；<https://huggingface.co/datasets/x-square-robot/xplanner-benchmark>（1,500 episodes / 3,490 videos）
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **核对范围：** 全文通读 arXiv `2609.25187v2`（§1–§6，含 Tables 1–3 与 Figs. 1–5）并逐文件对照官方仓库 `X-Square-Robot/Xplanner` 的训练、推理与评估脚本
- **通读：** 已完成
- **一句话说明：** 通过四层事件层级（L3–L0）和故障重规划语料构造可校验的任务规划数据集，基于 Qwen3.5-VL 设计离散 JSON 事件模式与并行的 Staircase 连续潜在 CoT 模式，并与下游世界–动作模型做特征桥接。

## 开源状态

- **部分开源（训练/V5 事件推理/基准已开源；存在外部数据后端依赖缺口且未包含潜推理 Staircase 与下游动作耦合权重）**（2026-11-14 源码核查）：代码（MIT）、`X-Planner-9B-0916` 权重（Apache-2.0）与 `xplanner-benchmark` 数据集均已公布，但 README 明确提示 `xDataset/main` 公开快照缺少 JSONL / Qwen3.5 处理器模块，`x_planner/` 内未包含论文中描述的 Staircase Decoding MoT 分支和下游世界–动作模型权重，真机实验仅公开静态汇总 JSON 指标而无可运行机器人控制/复现脚本。

## 核心摘录

### 1) 四层事件结构与故障恢复数据体系（§2–§3）

- **摘录要点：** 面向多源示范（Ego 含 L1–L3、UMI 含 L1–L3、遥操作 Teleoperation 含 L0–L3），将数据按 `L3 Task → L2 Subtask → L1 Action → L0 Segment` 结构化切分，并系统融入真实人工接管事件（takeover）与人造失败–重规划片段（覆盖 15 类失效模式）。公布的子集 benchmark 含 1,500 个长程片段（中位 5 个子任务 / 37.1 s，P90 达 15 个子任务 / 113.1 s）。

### 2) 离散事件规划与连续潜思维 Staircase Decoding（§4）

- **摘录要点：**
  - **离散事件状态模式（`L_SFT`）：** 将规划与在线监控拆分为 `initial_plan`、`ongoing`、`end`、`takeover`、`replan` 五类结构化输出协议，实时根据当前视觉状态判断是否完成子动作、维持原计划或生成重规划指令。
  - **连续潜推理 Staircase Decoding（`L_SFT + λ_CoT L_CoT`）：** 为避免逐字生成自回归长思维链的推理延迟，提出在中继深度 `N_r` 处分叉轻量 Mixture-of-Transformers（MoT）分支，单次前向并行吐出 `K_c` 个连续潜在 CoT token `ŷ_{1:K_c} = F_stair(x; N_r)`，并在训练期用冻结文本解码重构损失 `L_CoT` 锚定潜空间语义。
  - **规划–动作桥接（`L_align`）：** 无论离散文本还是连续潜表征，统一通过投影头对齐到下游世界–动作策略模型的文本条件空间 `c_l = M_T P_Q(H_q[m_txt])`。

### 3) 离线与真机双臂评测结果（§5, Tables 2–3 & Fig. 5）

- **摘录要点：**
  - **离线两步文本评测（Table 2，BERTScore-F1 / Doubao Judge Overall）：** `kimi3` 为 `0.9108 / 1.434`，**X-Planner** 为 `0.9011 / 1.411`，`Qwen` 为 `0.8878 / 1.367`，`Doubao` 为 `0.8846 / 1.366`。
  - **真机双臂 Reasoning Manipulation（Table 3，离散事件模式完整系统，Task Progress 0–100%）：** **X-Planner 均值 `71.60%`**，高于 `U-Scratch`（`59.50%`）、`π0.5`（`56.40%`）、`DreamZero`（`32.70%`）、`LingBot-VA`（`31.60%`）。
  - **真机泛化套件（Fig. 5，均值）：** **X-Planner `53.75`**，`DreamZero` `28.50`，`π0.5` `24.00`，`U-Scratch` `18.50`（`LingBot-VA` 为 N/A）。
  - **评测局限：** 论文未报告各任务误差棒；真机仅评测完整离散事件系统，并未在物理真机上做去除故障数据或对照连续 Staircase 潜模式 vs 离散模式的控制消融；训练使用真值历史，推理则喂入滚动模型自输出（存在 oracle-context 分布偏移）。

## 对 wiki 的映射

- [paper-x-planner-event-structured-task-planning](../../wiki/entities/paper-x-planner-event-structured-task-planning.md)
- [X-Planner 官方仓库归档](../repos/x_planner_event_structured_task_planning.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
