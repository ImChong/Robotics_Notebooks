# On the Opportunities and Risks of Frontier Models for 3D Modeling, Computational Design and Robotics

> 来源归档（ingest）

- **标题：** On the Opportunities and Risks of Frontier Models for 3D Modeling, Computational Design and Robotics
- **类型：** survey / research report / living document
- **年份：** 2026（September）
- **机构：** MIT CSAIL · Computational Design and Fabrication Group（CDFG）
- **作者：** Zhiyang Dou, Jamison Meindl, Akihisa Watanabe, Anna Deng, Tianyu Huang, Igor Sadalski, Harrison Liang, Minghao Guo, Benjamin Tod Jones, Wojciech Matusik（BibTeX 见项目页）
- **项目页：** <https://mit-cdfg.github.io/Survey-AI-for-3D-modeling-Robotics/>
- **GitHub companion：** <https://github.com/Frank-ZY-Dou/awesome-ai-3d-modeling-robotics>
- **入库日期：** 2026-09-29
- **一句话说明：** 以 **345 公开帖 / 243 case** 的 horizon scan 评估 GPT-6 Astra、Claude Opus/Fable、Gemini 等 **未微调 frontier agent** 在 Blender/CAD/仿真/真机上的能力：**3D/CAD 草稿化**、机器人 **离线写控优于在线闭环**、以及 **harness 与模型不可分** 三条主结论。

## 核心论文摘录

### 1) 方法论：演示即分布式用户研究

- **摘录要点：** 将社区 showcase 视为跨 Blender、FreeCAD/SolidWorks/CGM、Isaac/MuJoCo、真机的 **crowdsourced user study**；按 **Tier 1–3** 分层可验证性（代码 / 交互环境 / 仅演示片段）。
- **对 wiki 的映射：**
  - [frontier-models-3d-cad-robotics-survey](../../wiki/overview/frontier-models-3d-cad-robotics-survey.md) — 读法与局限。
  - [具身评测选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) — 「演示 vs 可复现 benchmark」。

### 2) Harness：Computer use / MCP / 机器人接口

- **摘录要点：** **Harness** = 连接模型与 CAD、DCC、仿真或机器人的软件层；**Computer use**（GUI 自动化）、**Tool server / MCP**（结构化 API）、**离线脚本导入** 暴露不同 action space 与反馈；Table 1 汇总 MCP（Blender/CGM/FreeCAD）、MecAgent（SolidWorks）、RoboDojo/ENPIRE 等。
- **对 wiki 的映射：**
  - [Open Code Review MCP 语境](../../wiki/entities/open-code-review.md) — MCP 作为 agent 工具层（非 CAD 专用）。
  - [RoboDojo](../../wiki/entities/robodojo.md) — 机器人 harness / 评测栈。

### 3) 能力：3D 与 CAD

- **摘录要点：** 3D：可编辑 **Blender 程序/scene**（非单块 mesh）；视频→场景 benchmark 上 GPT-6 Astra high **70** vs GPT-5.5 high **65**（Tang et al. 2026）。CAD：FreeCAD **100 任务** Parametric CAD Bench 平均 **~85%**（领先模型）vs GPT-5.6 Sol **~70%**；**公差 / 可制造性 / 物理零件** 尚未系统验证。
- **对 wiki 的映射：**
  - [状态估计 / 几何重建 hub](../../wiki/overview/hub-state-estimation.md) — real-to-sim 场景重建邻域。
  - [SimFoundry real-to-sim 场景生成](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md) — 仿真就绪资产线。

### 4) 能力：机器人（离线 vs 在线）

- **摘录要点：** **离线开发**（仿真写控制器再部署）最有效；**多秒级推理延迟** 阻碍快闭环。RoboDojo **42 任务** progress **1→29/100**（GPT-5.5→GPT-6 Astra）；真机粗放置 **19/20**，精细接触 sim **~4%**。危险物理指令拒绝率极低（GPT-6 Astra **2/100**）；有真机 campaign 因不安全动作损坏硬件而中止。
- **对 wiki 的映射：**
  - [GPT 6 Astra 具身策略评测](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md) — 独立 hybrid/direct 对照。
  - [foundation-policy](../../wiki/concepts/foundation-policy.md) — 通用模型作策略。
  - [VLA 方法页](../../wiki/methods/vla.md) — 端到端低层策略对照轴。

### 5) 评测与归因（§5.6）

- **摘录要点：** 固定 **task + harness + compute budget** 才能隔离模型进步；报告强调 benchmark 度量的是 **model+harness 系统**；应同时报告 **进度分、成本、失败尝试、物理验证**。
- **对 wiki 的映射：**
  - [hub-embodied-eval-benchmark](../../wiki/overview/hub-embodied-eval-benchmark.md)
  - [sim-vs-real-eval-gap](../../wiki/concepts/sim-vs-real-eval-gap.md)

### 6) 风险与建议（§7–8）

- **摘录要点：** 可靠性、延迟/成本、工程有效性（B-rep 合法 ≠ 可制造）、**物理安全**、出处与学分、教育评估方式；建议真机 **model-independent safety interlocks**、benchmark 发布 task/seed/log/grader。
- **对 wiki 的映射：**
  - [软件安全基础](../../wiki/concepts/software-security-basics.md) — agent 工具链风险邻域（非物理安全专页）。

## 开源核查（步骤 2.5，2026-09-29）

- **Companion 索引：已开源** — [Frank-ZY-Dou/awesome-ai-3d-modeling-robotics](https://github.com/Frank-ZY-Dou/awesome-ai-3d-modeling-robotics)。
- **243 case 中 43 有 runnable code** — 其余为演示或部分 artifact；各 case 外链仓库开源状态 **逐案** 以报告表格为准。
- **报告无单一「一键复现全部结论」仓库** — 复现应跟附录链接到 RoboDojo、Parametric CAD Bench、Inspect Robots 等 **独立** benchmark 仓。
