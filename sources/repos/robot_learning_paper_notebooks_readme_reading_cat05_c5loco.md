# Robot Learning Paper Notebooks README 与分类 05 清单通读

> 来源归档（原文通读 · 支撑 `wiki/overview/paper-notebook-category-05-locomotion.md`）

- **标题：** Robot_Learning_Paper_Notebooks（上游仓库 README + 站点首页分类 05 列表）
- **作者：** ImChong
- **类型：** repo / paper-notebooks / curated-list
- **原文：** <https://raw.githubusercontent.com/ImChong/Robot_Learning_Paper_Notebooks/main/README.md>；<https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html>（2026-11 抓取）
- **一句话说明：** 人形机器人学习方向的逐篇中文深读笔记站；首页按 14 类列出笔记，分类 05 为行走运动（Locomotion）。
- **核对范围：** 通读上游 README 全文（学习路线图、工程层补充、MimicKit 对照表、高影响力精选「行走经典」、笔记结构说明、项目结构）；通读站点首页分类 05「Locomotion」的全部论文标签（标题与日期，共 24 条）。未逐篇打开分类 05 下各论文的 HTML 深读笔记，也未读各篇 arXiv 原文。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **笔记结构（README）：** 基本信息 → 一句话总结 → 英文缩写速查 → 问题背景 → 方法详解 → 具体实例 → 工程价值 →（可选）MimicKit 源码对照 → 面试 Q&A。
- **路线图中的行走脉络（README）：** Domain Randomization (2017) → 四足地形 · 教师-学生 (2020) → Real-World Humanoid Locomotion (2023)（从四足换到全尺寸人形，因果 Transformer 读观测-动作历史直出关节目标）；四足地形 · 教师-学生 → OP3 Soccer (2023)；Domain Randomization → LCP (2024)（动作平滑，替代低通滤波）→ ASAP (2025)。
- **高影响力精选「行走经典」（README）：** Learning Quadrupedal Locomotion（arXiv:2010.11251）、Real-World Humanoid Locomotion（2303.03381）、Locomotion as Next Token Prediction（2402.19469）、Humanoid Parkour（2406.10759）、15-Minute Sim-to-Real（2512.01996）、ECO（2602.06445）。
- **站点分类 05（首页）：** 徽标 24 条，按 arXiv 日期新→旧；2026 年条目覆盖高速冲刺 / 急转弯（Push the Limit）、足底触觉（Tac4Loco）、人-人形步态生物力学对比、高平台攀爬（APEX）、能耗约束（ECO）、像素端到端（Now You See That）、感知（异构融合高度图 Hybrid Autoencoder、RPL、AME-2）、跨构型控制（XHugWBC）、历史条件 + 在线蒸馏（HoRD）、收缩映射嵌入（CMR）、楼梯奔跑（FastStair）、受限落脚点（PLANC）、轮滑（SKATER）、音频 / 视频驱动（Freestyle、RoboMirror）、自动奖励设计（E-SDS）、跨构型起身；较早条目为 Extreme Parkour (2023-09)、ANYmal Parkour (2023-06)、Learning to Walk in Minutes (2021-09)。
- **口径提示：** 首页分类 05 为 24 条，本库分类页索引 82 条（含「待深读」与跨类 / wiki 实体条目），数字须注明口径。

## 对 wiki 的映射

- [`wiki/overview/paper-notebook-category-05-locomotion.md`](../../wiki/overview/paper-notebook-category-05-locomotion.md)
- 站点归档：[humanoid-robot-learning-paper-notebooks](../sites/humanoid-robot-learning-paper-notebooks.md)
