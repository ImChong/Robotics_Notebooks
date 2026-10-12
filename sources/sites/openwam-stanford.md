# OpenWAM（Stanford）项目页

- **类型**：项目静态站点
- **收录日期**：2026-10-12
- **站点**：<https://openwam.stanford.edu>
- **论文**：<https://arxiv.org/abs/2610.07922>
- **代码：** <https://github.com/OpenWAM/OpenWAM>
- **文档：** <https://openwam.github.io/OpenWAM/quickstart/>
- **预训练权重：** <https://huggingface.co/OpenWAM-Stanford/OpenWAM-Pretraining>
- **沉淀到 wiki：** [paper-openwam-composable-world-action-models.md](../../wiki/entities/paper-openwam-composable-world-action-models.md)

## 开源核查（2026-10-12）

| 项 | 结论 |
|----|------|
| 代码 | **已开源**：训练、评测与视频→动作组合代码（仓库 AGPL-3.0-only，Alpha 阶段研究软件） |
| 权重 | **部分**：仅预训练视频模型权重在 Hugging Face；项目页未提策略 / 动力学检查点 |
| 数据 | LIBERO-Long-CF（32,000 个反事实片段，十个任务）列出但无下载链接；真机示教数据无链接 |
| 许可 | 项目页未写许可；仓库为 AGPL-3.0-only |

## 站点摘录要点

- 列出各预训练语料的轨迹数与小时数（Open X-Embodiment、AgiBot World Beta、Ego-Exo4D v2、InternData-A1、RoboCOIN、RoboMIND、FastUMI-100K、UMI 系列）。
- 评测框架：LIBERO 四套件每任务 50 episode（每套件 500，3 个训练种子）；LIBERO-90 迁移 4 个任务（64/74/21/45）；真机 2×Franka Research 3 的 Toast / Cube / Cups。
- 页面不作正式发布承诺，定位为「比较交互设计的共同试验台」。
- 注意：与 NUS 等的 OpenWAM（arXiv:2609.07398，站点 [openwam-official](openwam-official.md)）同名不同项目。
