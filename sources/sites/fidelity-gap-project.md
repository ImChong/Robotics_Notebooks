# fidelity-gap.github.io（Fidelity Gap 项目页）

> 来源归档（site / project-page）

- **标题：** What 30,000 Hours of Ego-centric Video Does Not Teach
- **类型：** site / project-page
- **官方入口：** <https://fidelity-gap.github.io/>
- **论文：** <https://arxiv.org/abs/2610.12464>
- **代码：** Soon；项目页未提供公开仓库链接（核查于 2026-10-13）
- **基准：** Soon；论文承诺公开 OOD 评测集，项目页尚无数据下载链接（核查于 2026-10-13）
- **入库日期：** 2026-10-13
- **一句话说明：** 展示 Cosmos 3 人类第一视角视频规模实验：手部/主体运动达到较高结构忠实度，但被操作物体的响应仍落后；项目页提供 SCS 交互式说明、数据规模结果和失败案例。

## 页面公开信息

| 资源 | 状态 |
|------|------|
| 项目页 | <https://fidelity-gap.github.io/> |
| arXiv | <https://arxiv.org/abs/2610.12464> |
| 论文 HTML | <https://arxiv.org/html/2610.12464v1> |
| 代码 | Soon；无公开仓库链接 |
| Benchmark | Soon；无可下载数据链接 |
| 训练视频数据 | 论文感谢 Mecka 提供；本文未提供可公开下载链接 |

## 核查摘要

- 项目页称 Edge（4B）、Nano（16B）、Super（64B）视频扩散模型在最多 30k 小时 egocentric 视频上训练；给一帧与手部动作，预测后续 16 帧（约 1.6 秒）。
- **SCS（Structural Consistency Score）** 分别测量手部和被操作物体的 mask IoU，聚焦动作是否在生成世界中产生正确结构变化；LPIPS 作为视觉感知质量控制指标。
- 骨架条件能用约 300 小时达到无骨架基线约 15k 小时的主体忠实度；对象忠实度增长慢得多。
- object-centric adaptive noise scheduling 通过动态区域图把训练信号转向交互物体；在 30k 小时下 object SCS 从 0.527 到 0.546，视觉质量基本不变，但差距仍在。
- 失败演示包括未拿起汤匙/纸条、接触失败、刚体变形、物体消失以及非刚性物体形变不合理。

## 开源核查

截至 2026-10-13，项目页的 **Paper Code** 与 **Benchmark** 都标为 **Soon**。论文提到会发布 OOD 评测集，但当前未提供下载入口。请勿将 arXiv 的 CC BY 4.0 论文许可误读为数据、代码或模型许可。

## 对 wiki 的映射

- [Fidelity Gap 实体页](../../wiki/entities/paper-fidelity-gap-egocentric-video.md)
- [Fidelity Gap 论文归档](../papers/fidelity_gap_arxiv_2610_12464.md)
