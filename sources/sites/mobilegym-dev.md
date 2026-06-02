# MobileGym 官网（mobilegym.dev）

> 来源归档

- **标题：** MobileGym — A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research
- **类型：** site（产品页 + 浏览器 Live Demo）
- **URL：** <https://mobilegym.dev>
- **论文：** <https://arxiv.org/abs/2605.26114>
- **代码：** <https://github.com/Purewhiter/mobilegym>
- **入库日期：** 2026-06-02
- **一句话说明：** 官方对外主页：TL;DR 指标、28 App 沙盒演示、架构图、Leaderboard、Sim-to-Real 与效率对比、一键 Live Demo。

## 页面要点（维护索引）

| 区块 | 内容 |
|------|------|
| Live Demo | 浏览器内启动模拟器（无需安装） |
| TL;DR | 416 模板、0 程序化裁判假阳/假阴（发布校验）、GRPO +12.8 pt sim / 95.1% 真机保留 |
| 28 Apps | 12 日常 + 16 系统；React/TS 替身，不连真实服务 |
| 三堵墙 | 不可读 / 不可重置 / 不可逆 → JSON 状态统一解决 |
| Leaderboard | 256 test 任务，Gemini 3.1 Pro 58.8% SR；L4 仍仅 21.9% |
| Sim-to-Real | 59 任务信号桶：sim +42.8 pt → real +40.7 pt |
| 效率 | ~400 MB vs ~4.5 GB/实例；~3 s vs ~78 s 冷启动 |
| Citation | BibTeX `wu2026mobilegymverifiablehighlyparallel` |

## 对 wiki 的映射

- 主实体：[MobileGym](../../wiki/entities/mobilegym.md)
- 论文：[mobilegym_arxiv_2605_26114.md](../papers/mobilegym_arxiv_2605_26114.md)
- 仓库：[purewhiter_mobilegym.md](../repos/purewhiter_mobilegym.md)
