# Robot Learning Paper Notebooks README 与分类 12 清单通读

> 来源归档（原文通读 · 支撑 `wiki/overview/paper-notebook-category-12-hardware-design.md`）

- **标题：** Robot_Learning_Paper_Notebooks（上游仓库 README + 站点首页分类 12「Hardware Design」列表）
- **作者：** ImChong
- **类型：** repo / paper-notebooks / curated-list
- **原文：** <https://raw.githubusercontent.com/ImChong/Robot_Learning_Paper_Notebooks/main/README.md>；<https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html>（2026-11 抓取）
- **一句话说明：** 人形机器人学习方向的逐篇中文深读笔记站；README 给出学习路线图与笔记结构，首页按 14 类列出笔记，分类 12 为硬件设计（执行器、灵巧手、腿 / 踝 / 头、整机平台、身体-控制协同设计等）。
- **核对范围：** 通读上游 README 全文（学习路线图中「状态估计、硬件与 Sim-to-Real 工程层」、工程层补充 day6、MimicKit 对照表、高影响力精选、笔记结构说明、项目结构）；通读站点首页分类 12「Hardware Design」下的论文标签列表（标题与 arXiv 日期）。未逐篇打开分类 12 各论文的 HTML 深读笔记，也未读各篇 arXiv 原文。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **笔记结构（README）：** 基本信息 → 一句话总结 → 英文缩写速查 → 问题背景 → 方法详解 → 具体实例 → 工程价值 →（可选）MimicKit 源码对照 → 面试 Q&A。
- **硬件在路线图中的位置（README）：** 硬件被放在横跨整条路线的「状态估计、硬件与 Sim-to-Real 工程层」；Berkeley Humanoid 定位为「硬件、执行器辨识与仿真共同设计；LCP 的部署平台之一」，ToddlerBot 为「校准、数字孪生、遥操作采集与部署工具链」。
- **工程层补充 day6（README）：** Berkeley Humanoid 笔记以 arXiv v1 为准、对照 ICRA 2025 版与官方训练代码；ToddlerBot 笔记以 arXiv v4 / CoRL 2025 为准、对照 v1 与官方代码 2.0；教学排序为 ASAP → BeyondMimic 之后、动作数据之前，首页 LCP → Berkeley Humanoid 连线标「部署平台」。
- **站点分类 12（首页）：** 徽标该类 21 条（与本库 39 条清单口径不同），按 arXiv 日期新→旧，主题涵盖：灵巧手（连杆驱动、可打印 PHD-Hand、MIDAS 低阻抗直驱手、MCR-Bionic、Bowden 腱驱手、Handroid）、低成本 / 开源整机（PhoneBot 复用智能手机、Fauna Sprout、ToddlerBot、Berkeley Humanoid）、踝 / 腿结构（双凸轮并联弹性踝、DecARt 解耦驱动腿、最优踝设计框架）、肌骨人形（肌肉特性与管理）、头部（DIJIT）、触觉手套（OSMO）、角色机器人（Olaf）、人级执行器（Human-Level Actuation）、身体-控制协同设计（Brain-Body Co-design 跌倒恢复、Embracing Evolution、Chin-Up 协同设计）。
- **口径提示：** 首页分类 12 为 21 条，本库分类页为 39 条（含待深读与跨类条目），数字须注明口径。

## 对 wiki 的映射

- [`wiki/overview/paper-notebook-category-12-hardware-design.md`](../../wiki/overview/paper-notebook-category-12-hardware-design.md)
- 站点归档：[humanoid-robot-learning-paper-notebooks](../sites/humanoid-robot-learning-paper-notebooks.md)
