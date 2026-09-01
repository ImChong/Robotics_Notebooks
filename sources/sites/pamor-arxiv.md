# arxiv.org/abs/2608.28213（PAMoR 官方预印本入口）

> 来源归档（ingest）

- **标题：** PAMoR: Parameterized Affective Motion Generation in Real Time for Humanoid Robots
- **类型：** site / project-page（arXiv 为截至入库日唯一官方入口）
- **官方入口：** <https://arxiv.org/abs/2608.28213>
- **入库日期：** 2026-09-01
- **一句话说明：** UCL Chengxu Zhou 组情感人形运动论文的 arXiv 预印本页；含方法图、V-A 平面示意、用户研究与 G1 真机部署描述；截至 2026-09-01 **未列代码仓库**。

## 页面公开信息（检索自 2026-09-01）

| 资源 | URL / 状态 |
|------|------------|
| arXiv 摘要 | <https://arxiv.org/abs/2608.28213> |
| arXiv HTML | <https://arxiv.org/html/2608.28213> |
| PDF | <https://arxiv.org/pdf/2608.28213> |
| **代码** | **未开源** — 摘要页 Code/Data 区无 GitHub / Hugging Face / 权重链接；无独立项目站 |
| 机构 | UCL Department of Computer Science |
| 资助 | ARIA grant SMRB-SE01-P06（论文致谢） |

## 与论文一致的公开主张（便于 wiki 溯源）

1. **V-A 参数化：** 从机器人运动学闭式计算效价（姿态扩张）与唤醒（运动能量），无需人工情感标注。
2. **可组合扩散：** 文本动作先验 \(D_\tau\) + 效价/唤醒先验 \(D_v, D_a\) 在共享 MVAE 潜空间逐步组合；动作与情感可独立在线编辑。
3. **真机：** 29-DoF Unitree G1 实时自回归生成 + SONIC 跟踪执行。
4. **感知：** 被试识别命令情感 Top-1 **0.384**，接近 Emilya 人体表演 **0.44**；显著优于 TextOp 情感 prompt 与 SMooDi 风格参考基线。

## 对 wiki 的映射

- [`wiki/entities/paper-pamor.md`](../../wiki/entities/paper-pamor.md)
- [`sources/papers/pamor_arxiv_2608_28213.md`](../papers/pamor_arxiv_2608_28213.md)
