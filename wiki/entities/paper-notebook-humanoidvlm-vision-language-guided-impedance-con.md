---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-26
arxiv: "2601.14874"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_humanoidvlm-vision-language-guided-impedance-con.md
summary: "HumanoidVLM 把\"挑阻抗参数 + 选抓取角\"这件老靠手调的事，外包给一个轻量管线：VLM 看一眼第一视角图把任务和物体说出来 → FAISS-RAG 从两个小数据库（9 个任务 + 9 个物体）里查出实验验证过的 stiffness/damping 与手指角→ 直接喂给 G1 的任务空间阻抗控制器，让接触富集的人形操作\"软硬合适\"。14 个测试场景命中率 93%。"
---

# HumanoidVLM

**HumanoidVLM: Vision-Language-Guided Impedance Control for Contact-Rich Humanoid Manipulation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

HumanoidVLM 把"挑阻抗参数 + 选抓取角"这件老靠手调的事，外包给一个轻量管线：VLM 看一眼第一视角图把任务和物体说出来 → FAISS-RAG 从两个小数据库（9 个任务 + 9 个物体）里查出实验验证过的 stiffness/damping 与手指角→ 直接喂给 G1 的任务空间阻抗控制器，让接触富集的人形操作"软硬合适"。14 个测试场景命中率 93%。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [人形论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation.html> |
| arXiv | <https://arxiv.org/abs/2601.14874> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_humanoidvlm-vision-language-guided-impedance-con.md](../../sources/papers/humanoid_pnb_humanoidvlm-vision-language-guided-impedance-con.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation.html>
- 论文：<https://arxiv.org/abs/2601.14874>

## 推荐继续阅读

- [机器人论文阅读笔记：HumanoidVLM](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation.html)
