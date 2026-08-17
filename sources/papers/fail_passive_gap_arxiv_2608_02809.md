# Fail-Passive Gap（arXiv:2608.02809）

> 来源归档（ingest）

- **标题：** Toward Certified Functional Safety for Industrial Humanoid Robots: The Fail-Passive Gap and a Feasibility Study
- **缩写：** Fail-Passive Gap（本文术语；无官方项目缩写）
- **类型：** paper / functional-safety / industrial-humanoid / certification / stop-category
- **arXiv：** <https://arxiv.org/abs/2608.02809>
- **HTML：** <https://arxiv.org/html/2608.02809>
- **PDF：** <https://arxiv.org/pdf/2608.02809>
- **提交：** 2026-08-03（arXiv v1，`cs.RO`）
- **项目页：** 无独立项目页（仅 arXiv）
- **代码：** 截至 **2026-08-17** 论文未列 GitHub / 数据集；检索无官方实现 → **确认未开源**
- **作者：** Caiwu Ding、Tao Cui、Lingyun Wang、Chengtao Wen
- **机构：** Siemens Foundational Technologies，Siemens Corporation（Princeton, NJ, USA）
- **入库日期：** 2026-08-17
- **一句话说明：** 工业人形的安全态是**主动平衡站住**，违反 ISO 13849-1 / EN 60204-1 的 fail-passive（断电即安全）假设；用已认证外部安全链当「量尺」，把不可认证残差钉在机侧反应链（SDA ↔ 平衡策略接口）。

## 核心论文摘录（MVP）

### 1) 问题与术语：fail-passive gap（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2608.02809>
- **核心贡献：** 固定臂 / cobot / AGV 的保护停是 Stop Category 0（切电，机器滑停到无害静止），可按 ISO 13849-1 / IEC 62061 / ISO 10218 / ISO 3691-4 评到 PL e / SIL 3。动态平衡双足**切电本身就是摔倒危害**；安全态是依赖实时策略的 **active safe state**。作者把这一认证空洞称为 **fail-passive gap**。
- **对 wiki 的映射：**
  - [Fail-Passive Gap 实体](../../wiki/entities/paper-fail-passive-gap.md)
  - [机器人安全状态机](../../wiki/concepts/robot-safety-state-machine.md)
  - [整机配电架构](../../wiki/concepts/robot-power-distribution-architecture.md)

### 2) 方法：用已认证外部链当仪器（§III–§V）

- **链接：** System Architecture + Safety Evaluation
- **核心贡献：**
  - 不做新安全硬件。在 \(3\,\mathrm{m}\times 1.5\,\mathrm{m}\) 半封闭 G1 EDU 抓放单元上搭 Detection–Evaluation–Reaction（D–E–R）：SICK deTec2 光幕 + 急停 + ET 200SP F-DI + SIPLUS CPU 1515SP PC2 F + 无线 PROFIsafe（SCALANCE W）。
  - 外部链闭合且可用 PFHD / DC / CCF / PL/SILCL 量化 → 剩下不可认证的只有**机侧反应链**。
  - Siemens S7-1500 急停参考例（Entry ID 21064024）：Reaction 能评 PL e，正因为是两只接触器 Stop Category 0。人形单元**故意没有接触器**，停靠无线安全报文 + 平衡站住。
  - 机载 Linux 上跑软件定义自动化（SDA / soft PLC，IEC 61131-3），与平衡策略同机。G1 计算硬件**非安全等级**，该端点**不是**认证 PROFIsafe F-host，不带 SIL/PL 声明；缺口被精确定位到 **SDA ↔ 平衡策略接口**。
- **对 wiki 的映射：**
  - [Fail-Passive Gap 实体](../../wiki/entities/paper-fail-passive-gap.md) — 流程总览
  - [系统工程知识链](../../wiki/overview/hub-systems-engineering.md)
  - [Unitree G1](../../wiki/entities/unitree-g1.md)

### 3) 人形特有安全分析（§V-F / §VIII-A）

- **链接：** Active Safe State
- **核心贡献：**
  - **摔倒即危害：** 过猛的保护停可能把扰动打出可捕获域，保护动作变成新危害；停必须是约束停（短距离且仍 capturable）。
  - **单支撑下界：** 双支撑可较快减速；单支撑必须先落当前步（capture point）再静站，故 \(t_{\mathrm{stop}}\) 有相位相关下界；ISO 13855 间距必须按最坏（单支撑）\(t_{\mathrm{stop}}\) 定。
  - **平衡策略残差风险：** 主动安全态要持续耗能与计算；策略故障 / 传感丢失 / 饱和没有 PFHD。文中对照学习式 safe-stoppability 监测（[arXiv:2603.22703](https://arxiv.org/abs/2603.22703)）：可缩小残差，但本身给不出可认证 PFHD/DC。
  - 间距 \(S=K\cdot T+C\)（ISO 13855）；\(T\) 被未认证 \(t_{\mathrm{stop}}\) 主导。
- **对 wiki 的映射：**
  - [Capture Point / DCM](../../wiki/concepts/capture-point-dcm.md)
  - [Balance Recovery](../../wiki/tasks/balance-recovery.md)
  - [Safety Filter](../../wiki/concepts/safety-filter.md)

### 4) 时序预算与实验结果（§VI–§VII）

- **链接：** Timing budget + Results
- **核心贡献：**
  - \(t_{\mathrm{response}}=t_{\mathrm{detect}}+t_{\mathrm{FDI}}+t_{\mathrm{scan}}+t_{\mathrm{PROFIsafe}}+t_{\mathrm{rx}}+t_{\mathrm{stop}}\)；每项标 [S] 规格 / [C] 配置 / [M] 实测。
  - 量级：光幕 11 ms；F-DI 4–10 ms；PLC 扫描 15–40 ms；PROFIsafe 30–39 ms（看门狗 \(\le 192\) ms）；机侧接收 5–20 ms；机械停 **0.3–1.0 s**。最坏约 **1.1 s**，由 \(t_{\mathrm{stop}}\) 主导。
  - 仅 \(t_{\mathrm{detect}}\ldots t_{\mathrm{PROFIsafe}}\) 落在可按常规评的外部链；\(t_{\mathrm{rx}}\)、\(t_{\mathrm{stop}}\) 在未认证反应链。
  - 通信丢失（S6，四种断法各 3 次）均在 **0.5–1.3 s** 内平衡站住、未摔倒。停距曲线、误触发率、可用性、丢包裕度标为后续工作。
  - **明确不宣称**端到端认证 PL e / SIL 3。
- **对 wiki 的映射：**
  - [控制环路延迟建模](../../wiki/formalizations/control-loop-latency-modeling.md)
  - [Fail-Passive Gap 实体](../../wiki/entities/paper-fail-passive-gap.md) — 实验与评测

### 5) 开源边界（步骤 2.5）

- **链接：** 无项目页；[arXiv:2608.02809](https://arxiv.org/abs/2608.02809)
- **核心贡献：** 可行性研究 + 认证缺口分析，硬件为西门子 / SICK 商用安全件 + 宇树 G1 EDU 专有平衡策略。截至入库日 **确认未开源（无可运行实现）**。
- **对 wiki 的映射：**
  - [Fail-Passive Gap 实体](../../wiki/entities/paper-fail-passive-gap.md) — 源码运行时序图不适用

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-fail-passive-gap.md`](../../wiki/entities/paper-fail-passive-gap.md)
- 互链参考：[机器人安全状态机](../../wiki/concepts/robot-safety-state-machine.md)、[整机配电架构](../../wiki/concepts/robot-power-distribution-architecture.md)、[Safety Filter](../../wiki/concepts/safety-filter.md)、[Capture Point / DCM](../../wiki/concepts/capture-point-dcm.md)、[Balance Recovery](../../wiki/tasks/balance-recovery.md)、[系统工程知识链](../../wiki/overview/hub-systems-engineering.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[控制环路延迟建模](../../wiki/formalizations/control-loop-latency-modeling.md)

## BibTeX

```bibtex
@misc{ding2026towardcertifiedfunctionalsafety,
  title         = {Toward Certified Functional Safety for Industrial Humanoid Robots: The Fail-Passive Gap and a Feasibility Study},
  author        = {Caiwu Ding and Tao Cui and Lingyun Wang and Chengtao Wen},
  year          = {2026},
  eprint        = {2608.02809},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2608.02809}
}
```
