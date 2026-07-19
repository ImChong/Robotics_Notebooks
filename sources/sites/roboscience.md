# RoboScience（en.roboscience.co）

- **类型：** 公司 / 通用具身智能（项目站）
- **收录日期：** 2026-07-19
- **主站：** <https://en.roboscience.co/>（中文：<http://www.roboscience.co/>）
- **说明：** 北京机科未来科技有限公司（2024-12 成立），自研 **Visics** 通用具身模型、**VLOA**（Vision-Language-Object-Action）大模型栈与 **RoboMirage** 可微物理引擎；强调跨本体灵巧操作与「物理引擎–仿真数据–端到端训练」闭环。

## 一句话

**以物体中心 3D 点云轨迹为中间表示，把具身世界模型（想象）与通用操作模型（执行）经 Object Trajectory 接口串成 VLOA 闭环，并以 RoboMirage + 互联网视频 + 仿真操纵数据双轨扩容。**

## 为什么值得保留

- **产业侧「级联世界模型 + 操作模型」样本**：与纯 2D 视频 WM、静态 3D 重建、端到端 VLA 并列，RoboScience 明确走 **3D 动态点轨迹** 第三路线，并与 **>1B 参数统一操作表示** 解耦硬件。
- **自研可微物理引擎 RoboMirage**：与 MuJoCo Warp / Newton / Genesis 等仿真栈对照，声称刚体–软体–铰接体统一接触、GPU 异构与 Pythonic API。
- **数据规模叙事**：博客披露 **>100 万小时** 物体中心操作视频（周增数十万小时）、**100 亿条** 全空间操纵仿真实例（2026 目标 1 万亿条）——可作为 **Scaling Law / 数据飞轮** 产业口径样本。
- **团队背景**：CEO 田野（中科大物理 + Stanford CME、前 Apple AI Platform 技术负责人）；首席科学家邵林（NUS 助理教授，Stanford 博后，D(R,O) Grasp / UniGrasp / SAM-RL 等）。

## 项目页核查（步骤 2.5 · 2026-07-19）

| 核查项 | 结论 |
|--------|------|
| **首页 / About / Blog / Footer** | 导航含 Blog、About、Recruiting、Cooperation；站点 CMS 有 **Download** 分类，但技术博文与首页 **无 GitHub / Hugging Face / Zenodo 链接** |
| **全站 HTML 检索**（`en.roboscience.co` 首页 + About） | `github` / `huggingface` / `open-source` / `code` **无匹配**（2026-07-19 `curl`） |
| **开放程度** | **未开源** — VLOA、Visics、RoboMirage、权重与训练代码均未公开 |
| **部分开放** | 英文站技术博客两篇（VLOA 世界模型 / 操作模型）+ 融资新闻 + 演示视频（CDN 托管）；**无论文 arXiv 链接** |
| **宣称将开源** | 截至入库日 **未见** "code will be released" 类表述 |

- **代码：** 截至入库日 **无官方仓库链接**
- **数据集：** **未公开**（仅博客披露规模目标）
- **模型 checkpoint：** **未公开**

## 公开信息要点

### 产品与技术栈

| 组件 | 角色 |
|------|------|
| **Visics** | 对外品牌下的通用具身 AI 模型（2026 发布叙事） |
| **VLOA** | Vision-Language-Object-Action 大模型；双引擎：**Embodied World Model** + **General Manipulation Model** |
| **RoboMirage** | 自研可微物理引擎：多体动力学、可扩展接触建模、GPU 加速、Python API |
| **Object Trajectory Interface** | 世界模型输出的 **3D 点云轨迹** → 操作模型的轨迹条件输入 |

### 融资与里程碑（About + 新闻）

| 时间 | 事件 |
|------|------|
| 2024-12 | 公司成立（田野创办） |
| 2025-11 | The Information **50 Most Promising Startups 2025** |
| 2026-02 | **Pre-A** 数亿元人民币（璞华资本领投等） |
| 2026-05 | VLOA 操作模型技术博文 Part 2 |
| 2026-06 | **A 轮** 10 亿元人民币；Visics 发布；邵林团队 ICRA 2025 操纵与运动最佳论文提名相关新闻 |

### 团队（About 页摘要）

- **田野** — 创始人兼 CEO；中科大物理学士、Stanford CME 硕士；前 Apple AI Platform 技术负责人。
- **邵林** — 联合创始人兼首席科学家、NUS 助理教授；Stanford 博士（Jeannette Bohg / Leonidas J. Guibas）；RSS 2023 最佳系统论文提名、ICRA 2025 机器人操纵与运动最佳论文奖相关荣誉。
- **刘鹏海** — 联合创始人；前科沃斯副总裁、电机总经理。
- **王涛** — 联合创始人；前商汤国香资本董事总经理。

## 交叉链接

- 技术博文 Part 1：[roboscience_vloa_embodied_world_model_part1.md](../blogs/roboscience_vloa_embodied_world_model_part1.md)
- 技术博文 Part 2：[roboscience_vloa_general_manipulation_part2.md](../blogs/roboscience_vloa_general_manipulation_part2.md)
- Wiki 实体：[roboscience-vloa.md](../../wiki/entities/roboscience-vloa.md)
