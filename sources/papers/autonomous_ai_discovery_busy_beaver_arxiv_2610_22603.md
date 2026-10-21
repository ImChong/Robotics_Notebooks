# At the Limits of Computability: Steps Towards Autonomous AI Discovery on the Busy Beaver Frontier

> 来源归档（ingest）

- **标题：** At the Limits of Computability: Steps Towards Autonomous AI Discovery on the Busy Beaver Frontier
- **类型：** paper / arXiv / AI research / formal methods
- **arXiv：** <https://arxiv.org/abs/2610.22603v2>
- **PDF：** <https://arxiv.org/pdf/2610.22603v2>
- **作者：** Marco Dos Santos, Vivek Veeriah, Alex Havrilla, Andrew Bolt, Goran Zuzic, Satinder Singh, Tom Zahavy
- **版本日期：** v1 于 2026-10-15 提交；v2 于 2026-10-19 修订
- **入库日期：** 2026-10-21
- **一句话说明：** 在 Busy Beaver (BB(6)) 未解决机器上，Gemini agent 流程自动提出图灵机执行加速规则，并通过 Lean 形式化证明其中 11 台机器的规则。

## 摘要级要点

- **研究问题：** 现有 AI 数学工作多解答人类预先提出的问题；本文探索 AI 能否在开放式数学领域中自主发现结构并提出可证明的猜想。
- **领域选择：** Busy Beaver 问题具有不可计算性；论文聚焦 BB(6) 中仍未解决的机器。
- **方法：** Gemini agent 流程结合模式识别、自动猜想、非形式化数学推理与 Lean 定理证明，发现描述图灵机多步执行的紧凑加速规则。
- **结果：** 系统为 BB(6) 中 11 台机器发现并形式化验证了 Collatz 风格的加速规则。作者暂称这些机器为 “Cryptids”；其停机行为仍未解决，该命名仍待社区接受。
- **边界：** 这是针对特定数学领域的自主发现案例，不是机器人实验，也不能单凭该结果推断通用科研 agent 已能独立选题或证明任意新定理。

## 核心论文摘录（MVP）

### 1) 从预定义问题转向自主发现

- **摘录要点：** 论文把问题从「AI 能否解答给定数学题」移到「AI 能否在开放领域中发现数学结构并形成新问题」。
- **对 wiki 的映射：** [AI Auto-Research（学术研究自动化）](../../wiki/concepts/ai-auto-research.md) — 区分问题生成、实验/推理执行与外部验证。

### 2) 发现与证明分层

- **摘录要点：** Gemini 流程负责模式识别、猜想生成和非形式化推理；Lean 将候选加速规则交由形式系统验证，使生成与证明具有不同的证据角色。
- **对 wiki 的映射：** [AI Auto-Research（学术研究自动化）](../../wiki/concepts/ai-auto-research.md) — 作为“生成快于验证”规律的一例，强调独立验证环节。

### 3) 结果的适用范围

- **摘录要点：** 11 台 BB(6) 机器获得精确、紧凑的执行加速描述及 Lean 形式化验证；这些机器的停机状态仍未解决，“Cryptids”是作者提出的暂定称呼。
- **对 wiki 的映射：** [AI Auto-Research（学术研究自动化）](../../wiki/concepts/ai-auto-research.md) — 记录狭义领域成果与不应外推的边界。

## 参考来源

- [arXiv 摘要与版本记录](https://arxiv.org/abs/2610.22603v2)
- [arXiv PDF](https://arxiv.org/pdf/2610.22603v2)
