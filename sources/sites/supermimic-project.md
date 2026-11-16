# SuperMimic 官方项目页

> 来源归档

- **标题：** SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model
- **类型：** site（论文项目页）
- **URL：** <https://supermimic.github.io/>
- **论文：** <https://arxiv.org/abs/2610.28480>（[sources/papers/supermimic_arxiv_2610_28480.md](../papers/supermimic_arxiv_2610_28480.md)）
- **机构：** 加州大学伯克利分校
- **墙钟时间：** 项目页另称 wall-clock 减少超过 50 倍（SuperMimic 单张 H100 约 3 小时；SONIC 128 张 L40 约 168 小时）；脚注给出的换算约 7200×（GPU-hours）与约 56×（wall-clock）（2026-10-22 核查）
- **官方代码：** 页面未列论文作者维护的 GitHub、Hugging Face 或其他代码/数据下载入口，亦未标注即将开源（2026-10-22 核查）
- **社区复现：** [Motphys/Uni-SuperMimic](../repos/supermimic-unilab.md) 与 [zita-ch/supermimic_vibed](../repos/supermimic-vibed.md) 均为独立复现，不是官方实现
- **入库日期：** 2026-10-22
- **再核日期：** 2026-11-11
- **研究团队公开实现（网站未直接链出）：** [HybridRobotics/supermimic](../repos/supermimic-hybridrobotics.md)（包含训练及回放入口；与论文全部实验的逐项对应关系未验证）
- **一句话说明：** 展示基于可微世界模型训练人形全身动作跟踪策略的方法、计算对比、仿真覆盖率与 Unitree G1 真机结果。

## 页面核查要点

- **方法：** 世界模型与策略在同一训练循环中联合更新；世界模型拟合刚体仿真器记录的转移，策略通过短时想象 rollout 获得解析梯度，梯度不穿过仿真器。
- **计算对比：** 项目页报告 1 张 H100 训练 3 小时，超过 SONIC 已发布 checkpoint 的归一化回报；SONIC 对照为 128 张 L40 训练 168 小时。完整训练循环计入 3 小时，checkpoint 评估不计入。
- **动作覆盖：** 同一 129,514 条训练动作评测集上，SONIC checkpoint 有 179 个失败，SuperMimic 单卡 3 小时有 25 个失败，双卡 9.8 小时为 0。页面注明该数字衡量训练动作覆盖，而非泛化。
- **真机演示：** 使用 Unitree G1；项目页称同一策略用于所有真机片段，无逐动作调参。
- **限制：** 截至核查日没有公开代码、数据或权重下载入口；不能据此判断完整复现路径。

## 独立代码复现

以下仓库在 2026-10-22 核查时已公开代码，但均自述为独立实现；官方项目页仍未提供作者维护的实现入口。

- [Uni-SuperMimic](../repos/supermimic-unilab.md) — Motphys 的 UniLab 复现，支持 mjwarp、MuJoCo、Motrix 后端。
- [supermimic_vibed](../repos/supermimic-vibed.md) — Isaac Lab 3 / MuJoCo-Warp 复现；README 明示其 AI 辅助快速生成且未经审查。

## 对 wiki 的映射

- [paper-supermimic](../../wiki/entities/paper-supermimic.md) — 论文与项目统一节点
- [sources/papers/supermimic_arxiv_2610_28480.md](../papers/supermimic_arxiv_2610_28480.md) — 论文原始资料归档

## 2026-11-11 补记：研究团队代码与 CoRL 视频入口

- 发现 [HybridRobotics/supermimic](../repos/supermimic-hybridrobotics.md)，其 README 自称基于 mjlab 的 SuperTrack 实现，代码有训练、回放、两个 G1 task。此前“页面无官方代码链接”是页面本身的历史观测，**不代表现今没有公开实现**。
- 官方 [CoRL 2026 Program](../sites/corl-2026-program.md) 作为直播/录像检索入口已入库；目前未核验到 SuperMimic 单场视频直链。


## 2026-11-13 会议获奖动态

- [11 月 12 日 CoRL 2026 最佳会议论文奖报道与核验边界](./corl-2026-supermimic-award.md)。新闻报告 SuperMimic 获 Best Conference Paper Award；Berkeley AI X 公告访问受限，项目页尚未更新奖项文字。本节是后续会议动态，不改变上述 2026-10-22 / 11-11 的项目页历史观察。
