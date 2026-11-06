# VABAR 项目页归档

> 来源：<https://berkeleyautomation.github.io/vabar-website/>（2026-11-06 抓取，HTTP 200）

- **标题：** VABAR — Variational Automation Benchmark for Agentic Robotics（UC Berkeley / NVIDIA / Siemens）
- **类型：** project-site
- **论文：** arXiv:2611.06881
- **入库日期：** 2026-11-06

## 站点要点

- 章节：Formulation、Criteria、Execution records、Tasks、Variation、Skills、Results、Diagnosis、Materials。
- **开放状态：** Repository `github.com/BerkeleyAutomation/VABAR` 标注 "available by Nov 9"；Materials 节承诺发布代码（Newton 与 ILA 两套实现的环境、实例生成器、评测器、基线运行器）、55+ 资产、33 技能与人工参考图、"N = 30 的种子实例集"。截至抓取日无许可证说明。
- **Simulation Platforms：** 链接 `robosimstudio/robosimstudio`（RoboSimStudio）与 `isaac-sim/IsaacLab-Arena`。
- **站点与论文的细微差异：** 表格列命名不同（站点 a Human-Newton / b Human-ILA / c Claude Code / d GaP，论文 a-i / a-ii / b / c）；站点称 ILA 列 N = 40 / 变体，论文正文只写 N = 30；站点 arXiv 链接仍是占位 `arXiv:XXXX.XXXXX`，BibTeX 年份 2027、键 `xie2027vabar`。
- **任务数字：** 站点各任务的 "Reference graph (Newton)" 数值与论文 Table II a-i 一致。
- **额外内容：** 站点有 syringe jam 的 "with / without unjam skill" 对比视频（无 unjam：验证报 blocked 并中止；有 unjam：push-and-re-verify 恢复并完成），论文正文未对该技能给出数字。

## 对 wiki 的映射

- [paper-vabar-variational-automation-benchmark](../../wiki/entities/paper-vabar-variational-automation-benchmark.md)
