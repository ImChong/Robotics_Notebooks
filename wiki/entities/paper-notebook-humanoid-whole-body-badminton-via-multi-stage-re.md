---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2511.11218"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_humanoid-whole-body-badminton-via-multi-stage-re.md
summary: "人形已能与静态场景交互（行走、操作），但动态实时交互仍难。作为迈向快速运动物体交互的一步，本文给出一条 RL 训练流水线，产出人形羽毛球的统一全身控制器，协调步法与击球，且不用动作先验、不用专家示范。训练遵循三阶段课程：① 步法获取；② 精度引导的挥拍生成；③ 任务聚焦精修——使腿与臂共同服务击球目标。部署时用扩展卡尔曼滤波（EKF）估计并预测羽毛球轨迹实现定点击球；并开发一个免预测变体（去掉 EKF 与显式预测）。仿真中双机可连续对打 21 拍；真机出球速度达 19.1 m/s、平均回球落点 4 米；EKF 版与免预测版表现相当。"
---

# Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning

**Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

人形已能与静态场景交互（行走、操作），但动态实时交互仍难。作为迈向快速运动物体交互的一步，本文给出一条 RL 训练流水线，产出人形羽毛球的统一全身控制器，协调步法与击球，且不用动作先验、不用专家示范。训练遵循三阶段课程：① 步法获取；② 精度引导的挥拍生成；③ 任务聚焦精修——使腿与臂共同服务击球目标。部署时用扩展卡尔曼滤波（EKF）估计并预测羽毛球轨迹实现定点击球；并开发一个免预测变体（去掉 EKF 与显式预测）。仿真中双机可连续对打 21 拍；真机出球速度达 19.1 m/s、平均回球落点 4 米；EKF 版与免预测版表现相当。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Whole-Body Badminton | 全身羽毛球，腿+臂协同击球 |
| Multi-Stage Curriculum | 多阶段课程，分步训练 |
| EKF | Extended Kalman Filter，扩展卡尔曼滤波 |
| Prediction-Free | 免预测，不显式预测球路的变体 |
| Footwork | 步法，移动到击球位 |
| Motion Prior | 动作先验，参考运动（本文不用） |

## 为什么重要

- **体育任务是动态交互的好试金石**：羽毛球要求步法+击球+预测三合一，区分度高；
- **课程学习可替代动作先验**：分阶段从零学复杂全身技能，减少对动捕的依赖；
- **「预测 vs 免预测表现相当」很有意思**：提示策略可隐式吸收球路规律；
- **与人形足球、拳击、网球等体育线互补**，共同推动高动态全身控制。

## 解决什么问题

人形与**快速运动物体**实时交互（如羽毛球）很难： - 要**协调步法与击球**（腿臂同服务于打到球）； - 既想**不依赖动作先验/示范**（难采、不泛化）； - 还要**预测球路**做定点击球。

论文要：一个**统一全身控制器**，从零（无先验）学会打羽毛球并真机可用。

## 核心机制

1. **无先验/无示范的统一羽毛球全身控制器**：三阶段课程让腿臂协同服务击球；
2. **EKF 球路预测**实现定点击球，并给出**免预测变体**（表现相当）；
3. **动态快速物体交互**：从静态场景迈向高速运动物体；
4. **真机实测**：出球 19.1 m/s、回球落点 4m、仿真 21 连拍。

方法拆解（深读笔记小节）：三阶段课程（无先验、无示范）；部署：EKF 球路预测 + 免预测变体；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Humanoid_Whole-Body_Badminton_via_Multi-Stage_Reinforcement_Learning/Humanoid_Whole-Body_Badminton_via_Multi-Stage_Reinforcement_Learning.html> |
| arXiv | <https://arxiv.org/abs/2511.11218> |
| 作者 | Chenhao Liu、Leyun Jiang、Yibo Wang、Kairan Yao、Jinchen Fu、Xiaoyu Ren |
| 发表 | 2025 年 11 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_humanoid-whole-body-badminton-via-multi-stage-re.md](../../sources/papers/humanoid_pnb_humanoid-whole-body-badminton-via-multi-stage-re.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Humanoid_Whole-Body_Badminton_via_Multi-Stage_Reinforcement_Learning/Humanoid_Whole-Body_Badminton_via_Multi-Stage_Reinforcement_Learning.html>
- 论文：<https://arxiv.org/abs/2511.11218>

## 推荐继续阅读

- [机器人论文阅读笔记：Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Humanoid_Whole-Body_Badminton_via_Multi-Stage_Reinforcement_Learning/Humanoid_Whole-Body_Badminton_via_Multi-Stage_Reinforcement_Learning.html)
