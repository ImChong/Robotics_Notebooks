# ScaleDP 项目页（scaling-diffusion-policy.github.io）

> 来源归档（site）

- **标题：** Scaling Diffusion Policy in Transformer to 1 Billion Parameters for Robotics Manipulation
- **类型：** site（论文项目页 + 视频/表格）
- **URL：** <https://scaling-diffusion-policy.github.io/>
- **论文：** [arXiv:2409.14411](https://arxiv.org/abs/2409.14411) — [`sources/papers/scaledp_arxiv_2409_14411.md`](../papers/scaledp_arxiv_2409_14411.md)
- **会议：** IEEE ICRA 2025（Accepted）
- **机构（页眉）：** 美的集团（Midea Group）；华东师范大学（East China Normal University）；上海大学（Shanghai University）；北京人形机器人创新中心（X-Humanoid / Beijing Innovation Center of Humanoid Robotics）
- **入库日期：** 2026-09-28
- **一句话说明：** 展示 **ScaleDP** 相对 **DP-T** 的架构改动（AdaLN + unmasking）、MetaWorld/真机实验与 **S→H 随参数量上升的成功率表**。

## 开源核查（步骤 2.5，2026-09-28）

| 资源 | 状态 | 说明 |
|------|------|------|
| 页内 **Code** 链接 | **误链** | `href=https://github.com/juruobenruo/DexVLA`（DexVLA 仓库，与本文无关） |
| 独立 ScaleDP / ScaleDP-H 仓库 | **未发现** | 全页除上述 DexVLA 外无 GitHub / HF / 数据链接 |
| 判定 | **未开源** | 以项目页实际链接为准；勿将 DexVLA 当作官方实现 |

## 公开要点（编译自项目页，2026-09-28）

### 架构（相对 DP-T）

1. **AdaLN 条件融合：** 观测与时间步 embedding 经 adaptive layer norm 调制动作 token，替代 cross-attention 融合（页面对比图）。
2. **Unmasking / 非因果自注意力：** 去噪时 action chunk 内 token 可互看，缓解「只执行第一步」的复合误差。

### 真机成功率表（项目页，%）

| 模型 | 单臂平均相关任务 | 双臂任务 | 总平均 |
|------|------------------|----------|--------|
| DP-T | 39.28±29.08 | （含于七任务） | 同上 |
| ScaleDP-H | 92.14±9.58 | 含 Bimanual Stack Cube 100 | 同上 |

（单任务明细：Close Laptop / Flip Mug / Stack Cube / Place Tennis / Put Tennis into Bag / Sweep Trash / Bimanual Stack Cube。）

### 与 arXiv 摘要差异（阅读注意）

- 项目页正文写「四真机任务相对 DP-T **+22.5%**」；arXiv 摘要写单臂 **+36.25%**、双臂 **+75%**——统计口径/任务子集可能不同，**以 PDF 实验节为准**。

## 关联资料

- 论文归档：[`sources/papers/scaledp_arxiv_2409_14411.md`](../papers/scaledp_arxiv_2409_14411.md)
- Wiki 实体：[`wiki/entities/paper-scaledp-scaling-diffusion-transformer-policy.md`](../../wiki/entities/paper-scaledp-scaling-diffusion-transformer-policy.md)
