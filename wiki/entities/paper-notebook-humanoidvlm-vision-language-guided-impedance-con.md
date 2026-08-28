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

**HumanoidVLM: Vision-Language-Guided Impedance Control for Contact-Rich Humanoid Manipulation** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

HumanoidVLM 把"挑阻抗参数 + 选抓取角"这件老靠手调的事，外包给一个轻量管线：VLM 看一眼第一视角图把任务和物体说出来 → FAISS-RAG 从两个小数据库（9 个任务 + 9 个物体）里查出实验验证过的 stiffness/damping 与手指角→ 直接喂给 G1 的任务空间阻抗控制器，让接触富集的人形操作"软硬合适"。14 个测试场景命中率 93%。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation.html> |
| arXiv | <https://arxiv.org/abs/2601.14874> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 结论

**HumanoidVLM 没有去学一个新的阻抗控制器，而是把「阻抗参数与抓取角怎么定」改写成一次检索问题：VLM 认场景、RAG 查已验证的参数，控制器本身保持不变。**

- 真正起作用的是两个很小的数据库（**9 个任务 + 9 个物体**）里**实验验证过的 stiffness/damping 与手指角**，由 FAISS-RAG 检索取出，直接喂给 G1 的任务空间阻抗控制器。
- 报告指标是 14 个测试场景 **93% 命中率**——衡量的是"参数查得对不对"，不是长时程操作的成功率，读数时别放大。
- 适用边界由数据库覆盖决定：任务或物体落在 9+9 之外时管线没有外推机制，这是最直接的失败模式。
- 定位是接触富集人形操作里替代手调参数的**轻量管线**，价值在工程可用而非方法新颖；本页仅索引级，机制细节待从深读笔记消化。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_humanoidvlm-vision-language-guided-impedance-con.md](../../sources/papers/humanoid_pnb_humanoidvlm-vision-language-guided-impedance-con.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation.html>
- 论文：<https://arxiv.org/abs/2601.14874>

## 推荐继续阅读

- [机器人论文阅读笔记：HumanoidVLM](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation/HumanoidVLM_Vision-Language-Guided_Impedance_Control_for_Contact-Rich_Humanoid_Manipulation.html)
