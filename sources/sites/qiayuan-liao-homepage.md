# Qiayuan Liao — 个人学术主页（一手资料归档）

- **类型：** 研究者个人主页 / 项目索引（非单篇论文、非单一代码仓库）
- **链接：** <https://qiayuanl.github.io/>
- **核查日期：** 2026-11-11
- **核对范围：** 逐节阅读主页研究者简介、Highlights、Research 六项与 Software 五项的完整公开正文及全部条目链接；交叉核查 Workhorse、SuperMimic、BeyondMimic、Berkeley Humanoid 官方项目页；阅读 whole_body_tracking 与 mjlab 仓库 README 的安装、训练入口与维护声明；核对 Walking in Narrow Spaces 的 arXiv 元数据及开源仓库 README。此处的“通读”仅指**本次 ingest 的主页原文和所列 README**，不代表逐篇通读全部六项研究论文。
- **通读：** 已完成
- **身份：** 主页自述为 UC Berkeley 三年级博士生，导师 Koushil Sreenath、Pieter Abbeel；此前在 Boston Dynamics 与 Scott Kuindersma 参与 E-Atlas 的强化学习控制。研究目标是通用人形机器人，聚焦敏捷鲁棒的全身移动操作、数据/硬件/控制协同设计。
- **对应 Wiki：** [Qiayuan Liao｜研究与开源索引](../../wiki/entities/qiayuan-liao.md)

## 主页研究条目（逐项对照）

| 主页条目 | 一手入口 | 主页所示发表 / 定位 | 本库既有对应 |
|---|---|---|---|
| SuperMimic | <https://supermimic.github.io/>；[arXiv:2610.28480](https://arxiv.org/abs/2610.28480) | CoRL 2026 Oral；单个可微世界模型训练人形全身控制策略 | [paper-supermimic](../../wiki/entities/paper-supermimic.md) |
| Workhorse | <https://hybridrobotics.github.io/workhorse/>；[arXiv:2610.09117](https://arxiv.org/abs/2610.09117) | 从人类数据学习视觉规划与全身跟踪 | [workhorse](../../wiki/entities/workhorse-humanoid-loco-manipulation.md) |
| BeyondMimic | <https://beyondmimic.github.io/>；[arXiv:2508.08241](https://arxiv.org/abs/2508.08241) | Science Robotics 2026；跟踪策略与扩散引导 | [beyondmimic](../../wiki/methods/beyondmimic.md) |
| MuJoCo Playground | <https://playground.mujoco.org/> | RSS 2025 Best Demo Paper；GPU MuJoCo 训练 | [mujoco-playground](../../wiki/entities/mujoco-playground.md) |
| Berkeley Humanoid: A Research Platform for Learning-based Control | <https://berkeley-humanoid.com/>；[arXiv:2407.21781](https://arxiv.org/abs/2407.21781) | ICRA 2025；面向学习控制的硬件平台 | [open-source-humanoid-hardware](../../wiki/entities/open-source-humanoid-hardware.md) |
| Walking in Narrow Spaces | [arXiv:2212.14199v3](https://arxiv.org/abs/2212.14199v3)；[实现](https://github.com/HybridRobotics/quadruped_nmpc_dcbf_duality) | IROS 2023；多面体避障约束的四足 NMPC + DCBF + WBC | [legged-control-ocs2（关联框架）](../../wiki/entities/legged-control-ocs2.md) |

## 主页软件条目与核验

| 项目 | 官方仓库 | 2026-11-11 核查结论 |
|---|---|---|
| whole_body_tracking | <https://github.com/HybridRobotics/whole_body_tracking> | **已归档（archived）**；README 同时标注 retired，原 BeyondMimic 跟踪训练实现转向 [mjlab tracking](https://github.com/mujocolab/mjlab/tree/main/src/mjlab/tasks/tracking)。旧版 README 可核对 `scripts/csv_to_npz.py`、`scripts/rsl_rl/train.py`、`scripts/rsl_rl/play.py`。 |
| legged_control | <https://github.com/qiayuanl/legged_control> | 主页标注 retired；GitHub 仓库未设 archived 标志，不能把“停止维护”和“GitHub 归档”混写。 |
| mjlab | <https://github.com/mujocolab/mjlab> | **公开且未归档**；README 列出 `uv run train Mjlab-Velocity-Flat-Unitree-G1` 和 `uv run train Mjlab-Tracking-Flat-Unitree-G1` 等实际入口。 |
| Berkeley-Humanoid-Lite | <https://github.com/HybridRobotics/Berkeley-Humanoid-Lite> | **公开且未归档**，主页宣称低于 5,000 美元的开源人形平台；这是独立 Lite 项目，不能自动等同 ICRA 2025 Berkeley Humanoid。 |
| MuJoCo Playground | <https://playground.mujoco.org/>；[源仓库](https://github.com/google-deepmind/mujoco_playground) | 主页标注 retired；与继续维护的 mjlab 不可混同。 |

## 来源勘误与开放边界

1. **Walking in Narrow Spaces 的作者不一致：** 主页该条复用了 Berkeley Humanoid 的作者列表。arXiv:2212.14199v3 的实际署名为 **Qiayuan Liao、Zhongyu Li、Akshay Thirugnanam、Jun Zeng、Koushil Sreenath**，以原论文为准。主页该条“Webpage”也指向 Berkeley Humanoid 页面，并非该四足论文主页。
2. **Berkeley Humanoid 源码死链：** 主页 ICRA 2025 条目的“Code”链接指向 <https://github.com/qiayuanl/isaac_berkeley_humanoid/>，本次核验返回 HTTP 404。仅凭主页不能认定该链接的代码仍公开；另一个 Berkeley-Humanoid-Lite 仓库不应被误认为其替代仓库。
3. **Workhorse：** 官方项目页截至本次检查仍明示 “The code is not released yet.”，不得标为已发布可运行实现。
4. **SuperMimic：** 个人主页和项目主页的直接入口为论文/演示；研究团队相关代码另见本库 [SuperMimic](../../wiki/entities/paper-supermimic.md) 的专项核验。不可将社区复现或仓库开源范围自动等同整篇论文的全部资产。
5. **论文状态：** 本次只索引主页并核验关键信息，不将主页全文阅读等同各论文全文阅读；详细方法、实验与局限应追溯各自原文归档。

## 推荐入口

- 人物：[Qiayuan Liao｜研究与开源索引](../../wiki/entities/qiayuan-liao.md)。
- 可运行的当前训练框架：[mjlab](https://github.com/mujocolab/mjlab)（以官方 README 与对应版本为准）。
- 旧版运动跟踪实现：[whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)（归档状态，不适合作为持续维护入口）。
