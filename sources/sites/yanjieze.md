# Yanjie Ze（迮炎杰）个人学术主页 — yanjieze.com

- **类型**：学者个人站点 / 履历、新闻与论文列表（原始资料归档）
- **收录日期**：2026-06-25
- **主链接**：<https://yanjieze.com/>

## 一句话

Stanford Computer Science 博士生（导师 Jiajun Wu、C. Karen Liu），本科上海交通大学（2023）；曾在 **Amazon FAR**（2025 夏秋季）与 **Figure AI Helix AI**（2025 冬季起）实习；研究主线为 **人形机器人通向具身通用智能的完整技术路径**——从 **运动重定向（GMR）→ 全身遥操作与跟踪（TWIST/TWIST2）→ 视觉 loco-manipulation（VisualMimic、ResMimic）** 的数据与控制闭环。

## 为什么值得保留

- **人形模仿学习簇的作者级索引**：站点集中列出 TWIST、TWIST2、GMR、VisualMimic、ResMimic、BEHAVIOR Robot Suite 等与仓库 [运动重定向](../../wiki/concepts/motion-retargeting.md)、[遥操作](../../wiki/tasks/teleoperation.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) 主线高度重合的工作入口。
- **职业节点可追溯**：新闻栏写明 **2026-01 入职 Figure AI Helix AI**、**ICRA 2026 Oral（TWIST2）**、**RA-L 2026（EgoNav）** 等时间锚点（以站点为准）。
- **开源软件总入口**：维护 [GMR](https://github.com/YanjieZe/GMR)、[TWIST](https://github.com/YanjieZe/TWIST)、[awesome-humanoid-robot-learning](https://github.com/YanjieZe/awesome-humanoid-robot-learning) 等社区高引用仓库。

## 履历要点（来自主页自述，2026-06-25 抓取）

- **当前**：Stanford CS PhD student；导师 Jiajun Wu、C. Karen Liu。
- **实习**：Figure AI Helix AI team（2025 Winter，站点新闻 **2026 Jan** 再次标注）；Amazon Frontier AI & Robotics / FAR（2025 Summer & Fall，合作 Rocky Duan、Guanya Shi、Pieter Abbeel）。
- **本科**：上海交通大学（2023）；合作 Xiaolong Wang、Huazhe Xu。
- **自述方向**：*We are building solid pathways on Humanoid Robots towards embodied general intelligence.*

## 新闻与时间线（主页 News 区块压缩）

| 时间（站点标注） | 事件 |
|------------------|------|
| 2026 Jun | EgoNav 被 RA-L 2026 接收 |
| 2026 May | TWIST2 入选 ICRA 2026 Oral |
| 2026 Jan | TWIST2、GMR（Retargeting Matters）被 ICRA 2026 接收；入职 Figure AI Helix AI |
| 2025 Dec | TWIST2 全栈开源 |
| 2025 Nov | TWIST2 发布 |
| 2025 Oct | ResMimic 发布 |
| 2025 Sep | TWIST 全栈开源；VisualMimic 发布 |
| 2025 Aug | GMR 全栈开源；TWIST、BEHAVIOR Robot Suite 被 CoRL 2025 接收 |
| 2025 Jun | TWIST 获 CVPR Workshop on Humanoid Agents **Best Demo Award**；iDP3、LCP 被 IROS 2025 接收 |

## 人形机器人主线论文与项目（按主页 Selected / Humanoid 区块归纳）

下列 **标题与官方链接以主页为准**；此处为 ingest 压缩索引，精读请以论文 PDF / 项目页为准。

| 主题侧重 | 代表工作 | 入口（示例） |
|----------|-----------|----------------|
| 人形导航 | EgoNav（Learning Humanoid Navigation from Human Data） | RA-L 2026；主页 Publications |
| 可扩展全身数据采集 | TWIST2 | <https://yanjieze.com/projects/TWIST2/> · [arXiv](https://arxiv.org/abs/2505.02833) · [代码](https://github.com/amazon-far/TWIST2) |
| 全身 loco-manipulation（残差学习） | ResMimic | 主页 Publications · [arXiv](https://arxiv.org/abs/2510.05070) |
| 通用运动重定向 | GMR / Retargeting Matters | [GMR 仓库](https://github.com/YanjieZe/GMR) · ICRA 2026 |
| 视觉全身 loco-manipulation | VisualMimic | 主页 Publications · [项目页](../../sources/sites/visualmimic-github-io.md) |
| 全身遥操作模仿 | TWIST | <https://yanjieze.com/TWIST/> · [代码](https://github.com/YanjieZe/TWIST) · CoRL 2025 |
| 家庭场景全身操作基准 | BEHAVIOR Robot Suite | 主页 Publications · CoRL 2025 |
| 3D 扩散 visuomotor | iDP3（Generalizable Humanoid Manipulation with 3D Diffusion Policies） | IROS 2025 Oral |

## 早期视觉 RL / 操作主线（SJTU 阶段，主页 All Papers 子集）

| 方向 | 代表条目 | 入口 |
|------|----------|------|
| 3D 表征 visuomotor | 3D Diffusion Policy（3D DP） | RSS 2024 Oral · [arXiv](https://arxiv.org/abs/2406.14627) |
| 视觉 RL + 自监督 3D | Visual RL with Self-Supervised 3D Representations | RA-L / IROS 2023 Oral |
| 灵巧操作视觉 RL | H-InDex | NeurIPS 2023 |
| 离线 RL + 预训练 LM | Unleashing the Power of Pre-trained Language Models for Offline RL | ICLR 2024 |

## 开源软件（主页 Software 区块）

| 项目 | 说明 | 入口 |
|------|------|------|
| GMR | 实时通用人形运动重定向 | [YanjieZe/GMR](https://github.com/YanjieZe/GMR) |
| TWIST | 全身遥操作模仿系统配套代码 | [YanjieZe/TWIST](https://github.com/YanjieZe/TWIST) |
| Awesome Humanoid Robot Learning | 人形学习论文与代码策展列表 | [YanjieZe/awesome-humanoid-robot-learning](https://github.com/YanjieZe/awesome-humanoid-robot-learning) |

## 对 wiki 的映射

- 升格页面：[wiki/entities/yanjie-ze.md](../../wiki/entities/yanjie-ze.md)
- 已有实体（作者一作或核心贡献）：[TWIST](../../wiki/entities/paper-twist.md)、[TWIST2](../../wiki/entities/paper-twist2.md)、[GMR 方法页](../../wiki/methods/motion-retargeting-gmr.md)、[VisualMimic](../../wiki/entities/paper-notebook-visualmimic.md)、[ResMimic](../../wiki/entities/paper-resmimic.md)
- 项目页归档：[twist2-project.md](./twist2-project.md)、[visualmimic-github-io.md](./visualmimic-github-io.md)
- 仓库归档：[awesome-humanoid-robot-learning.md](../repos/awesome-humanoid-robot-learning.md)

## 参考链接

- 个人主页：<https://yanjieze.com/>
- GMR 博客（站点 Talks）：<https://yanjieze.com/post/gmr/>
