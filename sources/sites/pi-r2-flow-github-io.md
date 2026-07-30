# πR² 项目页（pi-r2-flow.github.io）

> 来源归档（ingest 配套站点）

- **URL：** <https://pi-r2-flow.github.io/>
- **对应论文：** [πR²: Reactive Real-time Flow Policies](https://arxiv.org/abs/2607.26055)（arXiv:2607.26055，CMU）
- **代码：** <https://github.com/pi-r2-flow/pi-r2-flow>
- **入库日期：** 2026-07-30
- **一句话说明：** 官方落地页：问题演示（Sync / RTC / πR²）、双修改叙事、真机四任务视频与定量条形图、局限与 BibTeX。
- **开源状态：** **已开源**（页内与仓库互链；训练+部署齐全）。

## 页面要点（2026-07 快照）

| 区块 | 内容 |
|------|------|
| Headline | ~4× faster；plug & play finetune；真机最高 +30% |
| Method 1 | Proprioception-reactive diffusion forcing（快/慢通道） |
| Method 2 | Latency-adaptive staircase（拖动 \(d\) 演示） |
| Real world | Catch Book / Insert Box / Tidy Up Book / Don't Spill；相对 Sync / Naive Async / RTC |
| Limitations | 反应式数据难采；本体感偏重；仅局部重规划 |

## 开源核查（步骤 2.5）

| 项 | 状态（2026-07-30） |
|----|-------------------|
| 项目页 Code | **有** → GitHub `pi-r2-flow/pi-r2-flow` |
| 训练入口 | `learning/Isaac-GR00T` + `launch_finetune.py` 变体旗标 |
| 部署入口 | `deployment/apps/run_policy.py` |
| 结论 | **已开源** |

## 对 wiki 的映射

- 与 [sources/papers/pi_r2_arxiv_2607_26055.md](../papers/pi_r2_arxiv_2607_26055.md)、[sources/repos/pi-r2-flow.md](../repos/pi-r2-flow.md) 配对
- 实体页：[wiki/entities/paper-pi-r2.md](../../wiki/entities/paper-pi-r2.md)
