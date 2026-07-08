# LingBot-VLA 2.0（technology.robbyant.com）

> 来源归档

- **标题：** LingBot-VLA 2.0 官方项目页（Robbyant）
- **类型：** site
- **官方入口：** <https://technology.robbyant.com/lingbot-vla-v2>
- **代码：** <https://github.com/robbyant/lingbot-vla-v2>
- **技术报告 PDF：** <https://github.com/robbyant/lingbot-vla-v2/blob/main/assets/LingBot_VLA_2_0.pdf>
- **arXiv：** <https://arxiv.org/abs/2607.06403>
- **入库日期：** 2026-07-08
- **一句话说明：** Robbyant 对外产品化入口：预训练数据规模、双流过滤管线、55 维统一动作空间、MoE 与 Dual-Query 蒸馏可视化，以及 GM-100 / 长程移动操作真机 benchmark 与演示视频聚合。

## 策展要点（相对 README 的补充）

- **数据规模叙事：** 页面强调 **~50,000 h** 机器人真机 + **~10,000 h** embodiment-free egocentric manipulation 的混合预训练。
- **过滤管线图：** 灰带表示剔除阶段（video–state 错位、模糊遮挡、多视角错位、jerk/速度异常、静态片段；人侧 SLAM 不稳、手轨迹不连续等）。
- **演示任务：** 红豆倾倒、冰箱存食、炉灶清洁等 **高精度 / 铰接物体 / 长程移动操作** 真机片段。

## 对 wiki 的映射

- [LingBot-VLA 2.0](../../wiki/entities/lingbot-vla-v2.md) — 方法归纳以技术报告与仓库 README 为准，本页作官方导航与可视化锚点
