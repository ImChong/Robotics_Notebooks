# 161 篇人形 Loco-Manip 长文通读（第 03 组视觉感知视角）

> 来源归档（原文通读 · 支撑 `wiki/overview/loco-manip-161-category-03-visuomotor.md`）

- **标题：** 重磅整理！161篇论文带你看人形机器人移动操作的十个方向和技术版图全景
- **作者：** 具身智能研究室（微信公众号）
- **类型：** blog / survey-curation
- **原文：** <https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A>（发表 2026-06-26；本地全文抓取见 `sources/raw/wechat_humanoid_loco_manip_161_2026-06-26/`）
- **一句话说明：** 把 161 篇人形移动操作论文按"能力形成顺序"分十类；第 03 组收 37 篇，强调视觉、本体状态、场景理解与全身控制之间的闭环。
- **核对范围：** 通读本地全文去图后的正文：开篇十类定义与排序理由、十个分组导语、161 个条目标题；第 03 组 056–092 共 37 条的原文题目、项目链接、发表日期与"算法实现总结"逐条读完；其余 124 条的"算法实现总结"逐条浏览（部分超长行在终端显示时被截断，句尾"关键点"句式由全行统计补足）。未做的事：161 篇被综述论文的原文均未阅读，逐篇细节以各论文 arXiv 为准；161 张图片未逐张核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **第 03 组定位：** 开篇定义为"用视觉完成目标定位、场景理解和操作闭环"；组导语："强调视觉、本体状态、场景理解和全身控制之间的闭环"。
- **组成：** 056 AGILE 至 092 触摸梦（37 条），发表日期跨 2024-06（HYPERmotion）到 2026-06（EgoHumanoid、HALOMI、Humanoid Everyday 等）。其中 084–092 为中文意译题目条目（如 Humanoid Manipulation Interface、3D Diffusion Policies 即 iDP3 类工作、Choice Policies、DoorMan、Touch Dreaming）。
- **组内关键点句式分布（本次对 37 条统计）：** 扩散 / 流匹配条件生成 17；分层技能 / 专家路由 9；特权教师 -> 学生蒸馏 4；保留 VLM 语义并加状态与动作头（VLA）3；视频 / 世界模型先验 1；示范监督 + 动作 chunk 1；策略与控制同链路 1；AMP / DINO 视觉特征 1。
- **输入侧共性：** 几乎每条都以"相机图像 / 多视角观测"作为首要输入，常与本体状态、遥操作 / 外骨骼数据、人类视频 / 动捕、深度 / 点云 / 高度图或接触 / 触觉信号并用。
- **典型路线举例：** 数据闭环类（DemoHLM、HumanoidExo、HumanoidMimicGen、OASIS 等：遥操作或外骨骼采集 -> 模仿学习 / 分层技能 -> 动作命令）；教师-学生蒸馏类（SynAgent、VIRAL、VisualMimic、DoorMan）；VLA 类（EgoHumanoid、TrajBooster、WholeBodyVLA）；VLM 路由 + 技能库类（Being-0、HYPERmotion、HiWET）；3D / 深度感知类（OmniDP、3D 扩散策略、PhysHSI、抓取部署管线）；触觉 / 接触类（Touch Dreaming、HMC）。
- **同一工作跨组出现：** HMC、HiWET、VIRAL、WholeBodyVLA、DemoHLM、HumanoidExo 等同时出现在多个分组，分类是视角而非互斥划分。
- **局限（原文性质）：** "算法实现总结"为统一模板生成，句式高度重复，部分条目的机构栏混入摘要片段；不能替代论文原文。

## 对 wiki 的映射

- [`wiki/overview/loco-manip-161-category-03-visuomotor.md`](../../wiki/overview/loco-manip-161-category-03-visuomotor.md)
