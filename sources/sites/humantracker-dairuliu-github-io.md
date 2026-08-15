# dairuliu.github.io/humantracker（HumanTracker 项目页）

> 来源归档（ingest）

- **标题：** HumanTracker — Towards Comprehensive and Human-Aligned Motion Tracking Benchmark
- **类型：** site / project-page
- **官方入口：** <https://dairuliu.github.io/humantracker/>
- **入库日期：** 2026-08-15
- **一句话说明：** 论文配套站点：强调 **偏好对齐轨迹指标 HumanScore** + **四族诊断基准**；给出与 GMT / Humanoid-GPT / SONIC / TWIST2 的同步零样本视频网格。截至入库日，页头 Paper / Code / Dataset 仍写 **Coming Soon**，但页脚 BibTeX 已指向 arXiv:2608.13555，GitHub 仓已有可运行评测代码。

## 页面公开信息（检索自 2026-08-15）

| 资源 | 页头状态 | 实际可核 URL |
|------|----------|--------------|
| 项目首页 | 已上线 | <https://dairuliu.github.io/humantracker/> |
| 论文 | Coming Soon | <https://arxiv.org/abs/2608.13555>（BibTeX / arXiv HTML 已给） |
| 代码 | Coming Soon | <https://github.com/GalaxyGeneralRobotics/HumanTracker>（arXiv 页眉 `\code` 与仓库 README） |
| 数据集 | Coming Soon | 截至入库日无 HF / 下载链 |

## 与论文一致的公开主张（便于 wiki 溯源）

1. **评测应对齐人眼：** 运动学误差平均逐帧姿态差，漏掉支撑不稳、脚滑、接触时序和失败恢复。
2. **HumanScore：** 在测试集上与人类偏好对齐 **90.83%**，比最强常规诊断高 **6.78** 点。
3. **数据规模：** **153** 小时光学动捕、**25K** 标注 clip、**24** 名职业表演者、**4** 个诊断族。
4. **偏好数据：** 6 名专家标 **6,000** 原始对（严格 / 相似 / 不可比）；左右镜像得 **12,000** 条；每窗 **5 s / 250** 帧、**539** 维 token。
5. **可读输出：** 窗奖励经 sigmoid 后按真实帧数加权，映射到 **0–100**。
6. **族级规模（站点数字）：** Daily 89.29 h / 9,739；Highly Dynamic 11.01 h / 2,676；Interaction 47.78 h / 10,940；Ground 4.59 h / 1,640。
7. **对照 tracker：** 站点同步网格展示 GMT、Humanoid-GPT、SONIC、TWIST2 的零样本 rollout。
8. **指标分工：** HumanScore 看感知质量；Succ 看是否跑完；MPJPE 等仍用于定位具体误差——HumanScore **不是**所有解析诊断的替代品。

## 对 wiki 的映射

- [`wiki/entities/paper-humantracker.md`](../../wiki/entities/paper-humantracker.md) — 基准、HumanScore 与开源边界
- [`sources/repos/humantracker.md`](../repos/humantracker.md) — 官方评测仓
- [`sources/papers/humantracker_arxiv_2608_13555.md`](../papers/humantracker_arxiv_2608_13555.md) — 论文摘录
