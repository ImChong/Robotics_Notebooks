# markli1hoshipu/RoboHarness（RoboHarness 官方仓）

- **标题：** RoboHarness
- **类型：** repo
- **URL：** <https://github.com/markli1hoshipu/RoboHarness>
- **入库日期：** 2026-08-03
- **配套论文：** [arXiv:2607.18060](https://arxiv.org/abs/2607.18060) — [`sources/papers/robo_harness_arxiv_2607_18060.md`](../papers/robo_harness_arxiv_2607_18060.md)
- **项目页：** <https://www.robo-harness.com/> — [`sources/sites/robo-harness-com.md`](../sites/robo-harness-com.md)

## 一句话摘要

项目页标注的官方 Code 入口；截至入库日仓内以 **README + `docs/` 静态站点（HTML / 图 / 演示视频）+ `gen_lang_pages.py`** 为主，**尚未提供可运行的 harness 训练 / 推理 / 评测脚本**。

## 仓库核查（2026-08-03）

| 项 | 状态 |
|----|------|
| **公开可见** | 是（GitHub 公开仓，约 40 个文件树节点） |
| **许可证** | 未声明（API `license: null`） |
| **可运行入口** | **无** — 未见 `train` / `eval` / CLI package；`docs/` 为 GitHub Pages 风格项目页镜像 |
| **演示资产** | 有 — 仿真任务短视频、真机 `real_robot_demo.mp4`、框架与结果图 |
| **底层依赖（论文声明，非本仓）** | openpi π₀.₅；HF OpenVLA-OFT GRPO；PDDLStream / FF TAMP |

## 复现读法

- 需要 **端到端复现论文数字** 时：本仓目前 **不够**；应跟踪后续是否发布 harness 实现，并另接 openpi / RLinf 权重与自建 TAMP。
- 需要 **理解系统设计与演示** 时：README / 项目页已足够对齐 wiki 流程图与结果表。

## 关联资料

- 论文：[`sources/papers/robo_harness_arxiv_2607_18060.md`](../papers/robo_harness_arxiv_2607_18060.md)
- 项目页：[`sources/sites/robo-harness-com.md`](../sites/robo-harness-com.md)
- wiki：[`wiki/entities/paper-robo-harness.md`](../../wiki/entities/paper-robo-harness.md)
