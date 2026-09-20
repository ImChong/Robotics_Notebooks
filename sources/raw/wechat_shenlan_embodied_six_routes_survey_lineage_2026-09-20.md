# 六大具身路线详解：模块化、技能编排、IL、RL、VLA、世界模型，到底在"吵"什么。。。

> 原始抓取（WebFetch / mp.weixin.qq.com，2026-09-20；`--no-images` 等价：正文无图本地化）
> 来源：深蓝具身智能 · 《具身智能基础》专栏第 13 篇
> 编辑｜阿豹；审编｜具身君
> URL: https://mp.weixin.qq.com/s/iyzL2yLzIqsIRergN3qS_Q

---

大家好，这里是【深蓝具身智能】。本文出自《具身智能基础》专栏-第 13篇。

9 月的北京，有群背着相机和支架的年轻人，在老旧小区挨家敲门，说想进屋拍点东西。据说后来有老人报了警。

这不是诈骗，是数据采集。

几乎同一时间，一条在朋友圈传开的消息里，有人质疑圈内部分企业的「数采中心」，本质上是在把钱从左手倒到右手。

这两件事我不打算评对错。一场尚未被证实的指控，和一个刚起步就长出 70 多个训练场的产业，我们尚下不了定论。

但有几组数字，摆在一起看很有意思。

行业估算，要让一个通用机器人在常见任务上达到 70%–80% 的基础成功率，需要 1 亿小时。如果目标是真正开箱即用的通用具身智能，这个数字可能要爬到千亿小时。

与此同时，过去 18 个月全球冒出了 100 多款开源具身基础模型。到 8 月底，明确宣布做世界模型的国内公司至少 30 家。

现在，可谓是：全民抢数据，人人造模型。

所有人都在默认「只要数据够多，Scaling Law 就能在物理世界重现」。

但，机器人到底该从这些数据里学什么？

行业把「数据」当成了答案，而数据其实只是症状。

过去十年，这个行业对「机器人该从什么里学」给出过六次不同的回答：模块化规划、技能编排、模仿学习、强化学习、VLA、世界模型…

放下各种焦躁，可以发现今天具身智能试图解决的很多问题，其实都很'老'：

> 机器人怎么把一句任务拆成一连串动作？
> 怎么组合已经学会的技能？
> 怎样从人的示范中学习？
> 怎样通过试错掌握控制？
> 语言里的"杯子"、"拿起"、"放进去"，怎样和真实世界对应？
> 机器人能不能在行动之前，先预测一下接下来会发生什么？

这些核心诉求，从未因某个新"技术名词"的出现而消失。

因此，这篇文章打算沿着模块化规划、技能编排、模仿学习、强化学习、VLA、世界模型六条经典技术主线，结合近10年来12篇被高频引用的代表综述研究展开。

综述有个好处：路线争论会过时，benchmark 会被刷爆，但综述的分类法，和它当年点出来的未解难题，通常能活很久，很久。

## 模块化规划：先弄明白，机器人为什么总喜欢"拆开做"

（正文略：TAMP 形式化；2024 FM 综述按 perception / motion planning / control 看 FM 替换位置；层间翻译丢信息）

## 技能编排：机器人一直都有 Skill，变化的是"谁来安排它们"

（正文略：Behavior Tree 2022 综述；2024 LLM for Robotics；LLM 生成 PDDL + 经典规划器；BT vs LLM Planner）

## 模仿学习

（正文略：2009 LfD 综述；2026 FM for Manipulation 综述；示范从单次教学 → 基础模型训练资源）

## 强化学习

（正文略：2019 continuous control RL tour；2025 real-world DRL successes；WM 作 learned simulator）

## VLA

（正文略：2016 Symbol Emergence；2026 VLA survey；Grounding 十年跨度）

## 世界模型

（正文略：2020 MBRL survey；2026 WM for Robot Learning comprehensive survey；video WM）

## 收束

经典问题没有消失，变化的是解决单位越来越大。短期 VLA 仍是控制主体，WM 辅助规划/数据/仿真；长期双向融合——工程判断，非科学证明。

---

## 文内列出的 12 篇代表综述（标题级，入库日未逐条核 arXiv）

1. Integrated Task and Motion Planning.
2. Real-World Robot Applications of Foundation Models: A Review.
3. A Survey of Behavior Trees in Robotics and AI.
4. Large Language Models for Robotics: Opportunities, Challenges, and Perspectives.
5. A Survey of Robot Learning from Demonstration.
6. What Foundation Models Can Bring for Robot Learning in Manipulation: A Survey.
7. A Tour of Reinforcement Learning: The View from Continuous Control.
8. Deep Reinforcement Learning for Robotics: A Survey of Real-World Successes.
9. Symbol Emergence in Robotics: A Survey.
10. A Survey on Vision-Language-Action Models for Embodied AI.
11. Model-Based Reinforcement Learning: A Survey.
12. World Model for Robot Learning: A Comprehensive Survey.
