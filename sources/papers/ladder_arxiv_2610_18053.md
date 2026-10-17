# LADDER: A Long-Horizon Robot Manipulation Benchmark（arXiv:2610.18053）

> 来源归档（ingest）

- **标题：** LADDER: A Long-Horizon Robot Manipulation Benchmark
- **arXiv：** <https://arxiv.org/abs/2610.18053>（HTML：<https://arxiv.org/html/2610.18053>），2026-10-13 提交
- **项目页：** <https://ladder.insait.ai/>（归档：[ladder-project.md](../sites/ladder-project.md)）
- **代码（项目页列出，未能核实）：** <https://github.com/insait-institute/ladder>（归档：[ladder-insait.md](../repos/ladder-insait.md)）
- **Hugging Face 集合（项目页列出）：** <https://huggingface.co/collections/INSAIT-Institute/ladder-6abbbddc46691080f5d954c8>
- **作者：** Petko Petkov、Giuliano Albanese、Nikolay Nikolov、Luc Van Gool、Jonathan Tremblay、Danda Pani Paudel
- **机构（项目页）：** INSAIT；NVIDIA（Jonathan Tremblay）。Paudel 与 Tremblay 为共同指导
- **入库日期：** 2026-10-17
- **文章来源：** 具身智能小站 2026-10-17 公众号文章
- **一句话说明：** 同一环境内子目标数从 1 递增到 10 的长程操作基准，对比学习型低层策略与高层编排（文本子目标 / 工具选择），区分规划瓶颈与执行瓶颈。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 列出论文、GitHub、Hugging Face 集合与排行榜提交指南，未标注 "coming soon"。
- **仓库核查（2026-10-17）：** 用 WebFetch 访问 `github.com/insait-institute/ladder` 返回 HTTP 404；gh 因会话未授权该仓库返回 403。**仓库实际内容与可见性未能核实**（可能为私有或尚未公开）。
- **结论：** **项目页宣称已开放，但仓库内容未核实**；Hugging Face 集合同样未打开核验。

## 摘录 1：基准设计

- 145 个任务、八类（single destination、transfer、split destination、relay、revisit、exclusion、ordered、duplicates），1 至 10 个子目标；三个仿真套件 SimplerEnv-Long、RobotArena-∞-Long、RoboLab-Long；两种本体 WidowX 与 Franka；每个系统 2,024 个仿真 episode。
- 步数预算：WidowX 套件每子目标 100 步（5 Hz，20 s）；RoboLab-Long 900 至 4,950 步。
- 评测系统：学习型（低层策略、周期 / 本体 / 特权触发、本体验证器、VoLo）与工具选择型（CaP-X、TiPToP、PSP-Agent（作者提出）、GPT-Policy）。

**对 wiki 的映射：** [`paper-ladder-long-horizon-benchmark.md`](../../wiki/entities/paper-ladder-long-horizon-benchmark.md)

## 摘录 2：结果

- 低层策略：单子目标 85.4%，2 子目标 26.9%，5 子目标 0.0%，8–10 子目标 0.5%；总体 11.0%。
- 总体成功率（arXiv HTML）：特权验证器 29.6%、CaP-X 25.0%、TiPToP 9.4%、PSP-Agent 40.1%、GPT-Policy 62.0%。项目页给出最佳模型 63.3%（含 GPT-6 Astra 命名），与 HTML 的 62.0% 有出入，**口径差异未核实**。
- 离线 VLM 分解指令最高 93.1%，场景状态跟踪最高 70.8%；无验证时 32.6%–43.4% 子目标因编排器提前推进而失败，加验证后降到 3.4%。TiPToP 在 50.9% 子目标上找不到规划。
- 真机：低层策略多子目标 WidowX 1.2%、Franka 5.6%；WidowX 单目标单子目标 66.7% / 75.0% / 100.0%（低层 / 本体验证器 / PSP-Agent）；Franka 5 子目标 0.0% / 8.3% / 25.0%；仿真与真机对系统排序一致。
- 成本（项目页）：GPT-6 Astra 每完成一个任务 6.49 美元；Gemini 3.7 Flash 经 PSP-Agent 0.33 美元、完成率 42.8%。

**对 wiki 的映射：** 评测、结论小节。
