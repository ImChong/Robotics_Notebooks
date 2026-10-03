# REDACT（arXiv:2609.25450）

> 来源归档（paper）

- **标题：** REDACT: Robust Perceptive Locomotion under Unseen Visual Corruption
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.25450>
- **项目页：** <https://gatjungk.github.io/REDACT/>
- **项目网站仓库：** <https://github.com/gatjungk/REDACT>（网站、论文、图片和演示视频；不含方法实现）
- **作者：** Natapat Kirdwichai、Tobias Driskell-Poole、Andrei Sontea、Jadu Dash、Muhammad Burhan Hafez、Danesh Tarapore
- **机构：** University of Southampton
- **入库日期：** 2026-09-28；项目页核查更新：2026-10-04
- **一句话说明：** Teacher–Student + 持续特征遮蔽 + 共识门控：仅用干净仿真深度训练，提升未知视觉损坏下的四足感知运动鲁棒性。

## 开源状态

**研究实现未公开（2026-10-04 复核）。** 官方 GitHub 仓库 README 明确其内容是项目网站、论文、图片和演示视频；项目页公开架构与训练/校准参数，但未链接算法训练或部署代码。网站仓库未声明许可证。

## 核心摘录

1. **方法：** Teacher–Student；persistent feature masking；利用干净观测近似校准的 consensus gating。
2. **编码器：** 项目页展示残差深度编码器与 4×6 空间特征网格，结合 group normalization、谱范数约束和共享地形预测目标。
3. **门控校准：** 只使用干净 rollout；约 3,500 个 episode 每集采一帧；按 cell 99th percentile 阈值筛选，保留特征不足时启用回退。
4. **项目页评测摘要：** 未增广深度训练下，在页面列出的多类测试损坏和柱状障碍 sweep 中对比 REAL、Extreme Parkour；真机视频展示结构化障碍与森林地形零样本迁移。定量统计与协议请以论文为准。

**对 wiki 的映射**

- [REDACT 论文实体页](../../wiki/entities/paper-redact-robust-perceptive-locomotion.md)
- [项目页与资料状态](../sites/redact-gatjungk-github-io.md)
- [项目网站源码仓](../repos/gatjungk-redact.md)
- [策展周更](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
