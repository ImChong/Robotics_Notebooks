# Robot Learning Paper Notebooks README 与分类 04 清单通读

> 来源归档（原文通读 · 支撑 `wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md`）

- **标题：** Robot_Learning_Paper_Notebooks（上游仓库 README + 站点首页分类 04 列表）
- **作者：** ImChong
- **类型：** repo / paper-notebooks / curated-list
- **原文：** <https://raw.githubusercontent.com/ImChong/Robot_Learning_Paper_Notebooks/main/README.md>；<https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html>（2026-11 抓取）
- **一句话说明：** 人形机器人学习方向的逐篇中文深读笔记站，README 给出学习路线图、MimicKit 源码对照与高影响力精选；首页按 14 类列出笔记，分类 04 为运动操作与全身控制。
- **核对范围：** 通读上游 README 全文（学习路线图、工程层补充、MimicKit 对照表、高影响力精选、笔记结构说明、项目结构）；通读站点首页中分类 04「Loco-Manipulation and WBC」的论文标签列表（标题与日期）。未逐篇打开分类 04 下各论文的 HTML 深读笔记，也未读 progress.json 全表与各篇 arXiv 原文。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **来源与规则（README）：** 论文清单来自 YanjieZe 的 awesome-humanoid-robot-learning；笔记逐篇部署为网页，每篇结构为「基本信息 → 一句话总结 → 英文缩写速查 → 问题背景 → 方法详解 → 具体实例 → 工程价值 →（可选）MimicKit 源码对照 → 面试 Q&A」。
- **学习路线（README）：** PPO / AWR / Transformer → DeepMimic → AMP / PHC → ASE / CALM / PULSE / Diffusion Policy / BeyondMimic → 全身控制（Expressive WBC、HOVER、ExBody2、HugWBC、SONIC、TWIST）→ 分出操作、移动操作、世界模型三条上行支线 → VLA / BFM 终点。
- **移动操作支线（README ⑥）：** HOMIE → ULTRA → Ψ₀（外骨骼遥操作 → 多模态全身控制 → loco-manip 基础模型）；HOMIE → VIRAL / DoorMan（RGB 学生输出 HOMIE 控制器命令）；HOMIE ⇢ FALCON ⇢ CHIP（末端受力：抗扰补偿 → 可调柔顺）；DeepMimic ⇢ HDMI（从人类视频学交互，机器人与物体一起跟踪）。
- **全身控制核心精选（README）：** Expressive WBC、MaskedMimic、HOVER、ExBody2、UH-1、HugWBC、SONIC。
- **站点分类 04（首页）：** 首页徽标该类 75 条（与本库 142 条清单口径不同）；按 arXiv 日期新→旧，涵盖全身跟踪（OmniXtreme、EGM）、力自适应（FALCON、FAME、HAFO）、跌倒安全（SafeFall、VIGOR）、视觉移动操作（HERO、VIRAL）、人类视频 / 第一视角数据（HDMI、HumanX、EgoHumanoid、ZeroWBC）、语言驱动（FRoM-W1、SENTINEL）、体育技能（足球、羽毛球、拳击、网球）等主题。
- **口径提示：** README 徽标 Papers 1423 / Notes 418，首页分类 04 为 75，本库分类页为 142（含待深读与跨类条目），数字须注明口径。

## 对 wiki 的映射

- [`wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md`](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 站点归档：[humanoid-robot-learning-paper-notebooks](../sites/humanoid-robot-learning-paper-notebooks.md)
