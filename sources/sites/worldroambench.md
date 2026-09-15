# WorldRoamBench（项目页与在线榜单）

- **标题：** WorldRoam-Bench — An Open-World Benchmark for Long-Horizon Stability of Interactive World Models
- **类型：** site / benchmark portal
- **项目页：** <https://worldroam.amap.com/>
- **Leaderboard：** <https://worldroam.amap.com/#leaderboard>
- **打榜 / 提交：** 项目页「Submit Your Model」对话框（需按站点表单提交评测）
- **论文：** arXiv:[2606.31672](https://arxiv.org/abs/2606.31672)
- **机构：** 高德地图视觉技术实验室（Amap CV Lab）、南京大学（NJU）、清华大学（THU）、北京大学（PKU）
- **收录日期：** 2026-09-15

## 开源与数据（步骤 2.5，截至 2026-09-15）

| 资源 | 状态 | URL / 说明 |
|------|------|------------|
| 在线 Leaderboard | **已开放** | <https://worldroam.amap.com/#leaderboard> |
| 模型提交 | **已开放** | 站点「Submit Your Model」 |
| 评测数据集 | **已开放** | OSS：`https://amap-cvlab.oss-cn-zhangjiakou.aliyuncs.com/worldroambench/worldroam.zip`（约 2.5GB） |
| GitHub 评测代码 | **待发布** | 首页 GitHub 按钮标 `coming soon`，无公开仓库链接 |
| 论文 PDF | **已开放** | arXiv:2606.31672 |

## 站点要点（编译）

- **1000+ cases**，**4 维 15 指标**，评测 **10+** 开/闭源交互世界模型（Genie 3、HappyOyster、LingBot-World-V2、HY-World 1.5、Matrix-Game、SANA-WM、Lyra 2.0 等）。
- **对比表定位：** 相对 WildWorld / iWorld-Bench / WorldOlympiad / MIND / WBench / WorldMark，首个同时提供 **逐帧 action**、**visual drift**、**interaction physics**、**trajectory-aware memory** 的交互 WM 基准。
- **榜单更新截止：** 页面标注 Until 2026.08.20。

## 对 Wiki 的映射

- [paper-worldroambench](../../wiki/entities/paper-worldroambench.md)
- [worldroambench_arxiv_2606_31672.md](../papers/worldroambench_arxiv_2606_31672.md)
- [hub-embodied-eval-benchmark](../../wiki/overview/hub-embodied-eval-benchmark.md)

## 参考来源（原始）

- <https://worldroam.amap.com/>
