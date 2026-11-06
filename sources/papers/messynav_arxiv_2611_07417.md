# MessyNav（arXiv:2611.07417）

> 来源归档（paper）

- **标题：** MessyNav: Zero-Shot Navigation via Manipulation in Messy Environments
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.07417>
- **PDF：** <https://arxiv.org/pdf/2611.07417>
- **项目页：** <https://messynav.github.io/>
- **入库日期：** 2026-11-06
- **线索来源：** [只在虚拟房间练习，人形机器人走进陌生房间能做什么？](../blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)（具身智能小站，2026-11-06）
- **一句话说明：** 空推车可以挪，放着笔记本的推车却该绕开。

## 开源状态

- **待发布**（步骤 2.5 核查，2026-11-06）：项目页 HTTP 200，标注 Video and code coming soon。

## 核对范围与通读

- **核对范围：** 2026-11-06 通过 arXiv PDF（v1，CoRL 2026）通读正文第 1–6 节（含问题形式化、方法、三组实验、结论与局限）及附录 A（扩展相关工作）、B（硬件、三段 VLM 提示词、导航栈）、D.4（失败模式）、E（离线评测协议）；附录 C 的类别清单与 Table 6/8 场景明细仅浏览，未逐行核对。
- **通读：** 已完成

## 核心摘录（论文原文核对）

- **问题：** 开放世界交互导航（open-world interactive navigation）：给定第一视角图像与导航目标，机器人要判断是否需要交互、如何交互，并在新物体、新环境、新语境三个轴上零样本泛化。
- **方法：** 三个模块——Affordance Reasoner（VLM 先判断机器人可行性：负载 3 kg、单臂二指夹爪；再判断物体交互安全：易碎、昂贵、会洒落、排列有意图等；两关都过才标 interactive，属性为 safe/fragile/immovable，功能相关物体合并为一个整体）；Task Planner（输入标签+图像，输出 move_to / pick_up / swipe_left / swipe_right 计划）；Open-Language Skill Library（VLM 指点与分割，深度反投影到世界系，Nav2+LiDAR 避障）。
- **机器人：** TidyBot++（7 自由度 Kinova 臂+全向底盘）、Unitree L2 LiDAR（FAST-LIO）、两个 ZED 2 相机；仿真之外无训练，推理与规划用 gemini-3-flash-preview（thinking=minimal），指点/分割用 gemini-2.5-pro。
- **表 1 可供性分类（100 图 217 物体，14 地点，94 交互/123 非交互）：** MessyNav 85.7%；height-only 59.9%；closed-set 24.4%；closed-set-oracle 66.4%。换用 claude-opus-5 / gemini-3-flash-preview / gpt-5.6-luna 分别为 87.1% / 85.7% / 86.2%（oracle 69.1% / 66.4% / 67.3%）。
- **表 2 真机（15 场景，2 环境，4 方法共 60 次）：** MessyNav 67%（10/15，45.8 s）；height-only 13%（2/15，24.5 s）；vlm-planner-only 40%（6/15，45.9 s）；closed-set-planner 27%（4/15，50.4 s）。成功要求到达目标且无不安全交互；摘要所称 +27 pp 即 67%−40%。
- **表 3 离线规划（55 场景 5 环境，15 轮，每方法 825 次，假设低层完美执行）：** MessyNav 62.4% vs vlm-planner-only 55.4%（+7.0 pp，配对 t 检验 t(14)=4.71，p<0.001，95% CI [+3.8,+10.2] pp）；户外白天 78.3 vs 60.8，室内亮 62.8 vs 55.5，夜间 29.2 vs 44.2（MessyNav 反而更差）。
- **失败分析（附录 D.4）：** vlm-planner-only 9 次失败中 7 次是规划失败（动了不该动的物体），MessyNav 仅 2 次；MessyNav 余下失败多来自 Gemini 指点错误（如指到推车内部空地）。
- **局限（原文）：** 原语开环执行；受限于 VLM（弱光、从 2D 推几何）；无持久记忆，难以长程规划与恢复（如记住背包太重）。

## 对 wiki 的映射

- [paper-messynav](../../wiki/entities/paper-messynav.md)
- [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-06-20-papers-technology-map.md)
