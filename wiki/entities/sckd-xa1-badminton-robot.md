---
type: entity
project_id: sckd-xa1-badminton-robot
tags: [sports-robotics, badminton, mobile-robot, commercial-robot]
status: draft
updated: 2026-10-16
project: "https://www.zjsckd.com/product/"
related:
  - ./paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re.md
  - ./paper-humanoid-badminton-dynamic-racket-skills.md
sources:
  - ../../sources/sites/sckd-xa1-badminton-robot.md
summary: "浙江深辰凯动 AI 羽毛球机器人：以移动底盘、来球感知与击球执行构成商业陪练产品；Guinness World Records 记载 XA1 于 2025-12-22 连续回球 1,452 次。"
---

# 深辰凯动 AI 羽毛球机器人（XA1）

**深辰凯动 AI 羽毛球机器人**是面向羽毛球对打与陪练场景的移动机器人产品。Guinness World Records 记载，名为 **XA1** 的机器人于 2025 年 12 月 22 日在绍兴完成 1,452 次连续回球；公司产品页则以“X 型羽毛球机器人”介绍产品。公开页面尚未说明两个称谓的具体型号映射。

## 一句话定义

一款把移动接球、来球识别和球拍击球整合起来的商业羽毛球陪练机器人，已通过人机连续对打挑战展示长期稳定回球能力。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| AI | Artificial Intelligence | 产品方对机器人智能感知与决策能力的统称 |
| GWR | Guinness World Records | 认证并发布 1,452 次连续回球纪录的机构 |
| CV | Computer Vision | 该产品所需的来球视觉感知环节；官方称其采用高速视觉追踪 |

## 为什么重要

- 羽毛球对打把高速运动目标感知、落点应对、移动和击球时序放在同一闭环中，是动态机器人交互的直观测试场景。
- 连续 1,452 次回球的纪录提供了一个商业机器人耐久交互案例，可与研究原型的短回合演示和实验指标对照阅读。
- 产品定位覆盖职业与青训陪练、校园体育教育及公众体验，显示体育机器人从实验室验证走向场景化产品的一种路径。

## 项目概览

| 项目 | 信息 |
|---|---|
| 产品方 | 浙江深辰凯动科技有限公司 |
| 产品称谓 | 公司页面称“AI 羽毛球对打机器人 / X 型羽毛球机器人”；吉尼斯纪录页称创纪录机器人 XA1 |
| 纪录 | 连续回球 1,452 次 |
| 时间与地点 | 2025-12-22，中国浙江绍兴 |
| 公开用途 | 职业与青训陪练、学校体育教育、展馆和公众体验 |
| 公开资料 | 官方产品页、公司介绍、Guinness World Records 纪录条目 |

## 系统与能力

从任务闭环看，该产品需要识别来球、估计击球机会、移动到底盘可达区域并执行回球。深辰凯动官方页面将高速视觉追踪、自主运动控制、高动态移动和精准击球列为系统能力；公开资料没有披露其传感器布局、球路估计方法、规划控制结构或实时软件接口，因此这里只描述可确认的系统边界，不推断具体算法。

产品页还宣称来球识别与判断用时为 0.5 毫秒、回球成功率超过 99%。由于公开页面未列出指标定义和试验条件，这些数值应视为厂商公开口径，不能与吉尼斯纪录的计数结果混为同一类评测。

## 工程实践

- **真实场景关注点：** 连续回球要求感知延迟、底盘到位时间、击球窗口和整机可靠性同时满足；纪录挑战主要说明系统完成了长序列人机对打。
- **开源与复现状态：** 截至 2026-10-16，官方产品页和公司介绍未列代码、数据、权重或技术论文入口；当前公开材料不足以复现产品控制系统。
- **源码运行时序图：** **不适用。** 官方页面未提供可运行代码或模块接口，无法按源码还原运行时序。
- **命名边界：** 公司页面使用“X 型羽毛球机器人”，纪录页使用“XA1”；在官方澄清前，不把两个称谓扩写成更多产品型号。

## 实验与纪录

Guinness World Records 记载该机器人与人类对手连续完成 **1,452 次**羽毛球回球，挑战于 2025-12-22 在绍兴进行。公司报道补充对手包括前羽毛球世界冠军卢兰与职业选手朱思。纪录验证的是特定挑战规则下的连续回球次数；它不等同于回球成功率、比赛胜率或不同场馆条件下的泛化表现。

## 局限与风险

- **技术透明度有限：** 传感器、运动学、决策控制、硬件规格及接口细节未在当前官方资料中公开。
- **指标口径需区分：** 1,452 次是纪录挑战结果；产品页“99% 以上回球成功率”未附公开测试方法，二者不能相互替代。
- **开源复现入口缺失：** 当前未找到官方代码、数据集、权重或论文，外部开发者无法依据现有页面重建系统。
- **型号表述仍待核实：** 官方产品页与纪录页使用不同称谓，页面没有说明 XA1 与“X 型”之间的关系。

## 关联页面

- [Humanoid Whole-Body Badminton：无人类动作先验的退火 RL](./paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re.md) — 人形机器人全身羽毛球策略，可作研究型系统对照。
- [Humanoid Badminton：从有限人类动作学习动态球拍技能](./paper-humanoid-badminton-dynamic-racket-skills.md) — 研究人形机器人多技能回球与人机连续对打。

## 参考来源

- [项目与纪录官方资料归档](../../sources/sites/sckd-xa1-badminton-robot.md)
- [深辰凯动 AI 羽毛球机器人产品页](https://www.zjsckd.com/product/)
- [Guinness World Records 纪录条目](https://www.guinnessworldrecords.com/world-records/781002-most-consecutive-badminton-counter-hits-with-a-mobile-robot)

## 推荐继续阅读

- [深辰凯动官方产品页](https://www.zjsckd.com/product/)
- [连续羽毛球回球纪录详情](https://www.guinnessworldrecords.com/world-records/781002-most-consecutive-badminton-counter-hits-with-a-mobile-robot)
