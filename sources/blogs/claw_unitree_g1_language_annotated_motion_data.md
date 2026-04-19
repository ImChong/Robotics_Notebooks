# claw_unitree_g1_language_annotated_motion_data

> 来源归档（blog / wechat）

- **标题：** 动捕服可以扔了？南大×伯克利新研究宇树CLAW：为宇树G1生成物理仿真数据，比动捕更丝滑！
- **类型：** blog
- **来源：** 微信公众号文章（Yuanxq）
- **原始链接：** https://mp.weixin.qq.com/s/MNwq3k8MiNHMLuleDyFiHw?poc_token=HEZG5GmjQ4vz-AX9DP8Ok8m-sV4FonbPeeGbhgUw
- **入库日期：** 2026-04-19
- **最后更新：** 2026-04-19
- **一句话说明：** 这篇文章介绍了 CLAW：一个面向宇树 G1 的网页交互式数据生成管线，用物理仿真替代传统动捕，自动生成带语言标签的全身运动数据。

## 核心摘录

### 1) CLAW: Composable Language-Annotated Whole-Body Motion Data Generation for Humanoid Robots
- **文章作者：** Yuanxq
- **文中论文标题：** CLAW: Composable Language-Annotated Whole-Body Motion Data Generation for Humanoid Robots
- **文中发布时间：** 2026年4月
- **核心要点：**
  - 用网页交互界面替代真人动捕服 + 动捕棚，操作者可以通过键盘控制或时间轴编辑来组合宇树 G1 的全身动作。
  - 系统把走路、下蹲、匍匐、拳击等基础运动模式当作可组合原子动作，再通过底层规划器生成连续轨迹。
  - 通过模板化语言注释引擎，自动把动作状态翻译成带时间信息的文本描述，形成“动作轨迹 + 语言标签”配对数据。
  - 后端依托 MuJoCo 物理仿真和底层控制器，保证输出的关节轨迹符合物理约束，减少传统 mocap 重定向常见的滑步、穿模和不自然切换。
  - 该流程本质上把“人力密集的数据采集问题”转成“算力密集的数据生成问题”，适合为 humanoid 的语言-动作对齐、模仿学习和全身控制训练提供规模化数据。
- **关键洞见：**
  - 对人形机器人来说，数据瓶颈往往不只是“量不够”，而是“动作质量、物理一致性和语言标签难同时拿到”。
  - CLAW 的价值不在于提出一个新控制器，而在于提供了一条更便宜、更可扩展的 humanoid whole-body motion 数据生产管线。
  - 这类系统特别适合作为 imitation learning / language-conditioned policy / foundation policy 的前端数据引擎。
- **对 wiki 的映射：**
  - [imitation-learning](../../wiki/methods/imitation-learning.md)
  - [motion-retargeting](../../wiki/concepts/motion-retargeting.md)
  - [foundation-policy](../../wiki/concepts/foundation-policy.md)
  - [loco-manipulation](../../wiki/tasks/loco-manipulation.md)

## 原文简摘

- 文章指出，传统 humanoid 动作数据采集依赖昂贵动捕影棚，且跨骨骼重定向容易带来滑步和穿模。
- CLAW 用浏览器交互（键盘 / 时间轴编辑器）驱动宇树 G1 在 MuJoCo 中生成全身动作轨迹。
- 语言注释不是后处理人工标注，而是根据底层已知动作模式、速度和时序自动模板化生成。
- 输出的 50Hz 关节序列经过物理约束校验，可直接作为后续训练或部署前的高质量数据源。

## 当前提炼状态

- [x] 文章基础摘要填写
- [x] 初步 wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接
- [ ] 若后续需要，可继续把文中 CLAW 拆成独立概念页或 query 页
