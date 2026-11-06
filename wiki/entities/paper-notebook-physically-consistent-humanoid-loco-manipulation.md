---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2504.16843"
venue: "2025.04"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_physically-consistent-humanoid-loco-manipulation.md
summary: "Physically Consistent Humanoid Loco-Manipulation using Latent Diffusion Models：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# Physically Consistent Humanoid Loco-Manipulation using Latent Diffusion Models

**Physically Consistent Humanoid Loco-Manipulation using Latent Diffusion Models** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Physically Consistent Humanoid Loco-Manipulation using Latent Diffusion Models 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在深读笔记写出来之前，这里只保留分类位置与原文入口；方法流程见「流程总览」（依据论文方法部分），结论与指标以论文为准。

## 流程总览

```mermaid
flowchart TD
    IN["输入: 高层任务指令 P (文本提示序列，可来自语言模型)<br/>+ 场景物体 RGB-D 图像"] --> GEN
    subgraph PL["规划接触与机器人构型 §III-A"]
        GEN["1. 图像生成: 提示词追加全身人物描述<br/>LDM 生成人-物交互 RGB 图像"]
        GEN --> CT
        subgraph CT["2. 接触迁移 Contact Transfer"]
            M1["VLM 开放词汇检测 + 分割基础模型 → 物体掩码"] --> M2["深度基础模型估计生成图深度 → 2D 到 3D 点云 + 去噪"]
            M2 --> M3["语义匹配基础模型给候选对应<br/>随机采样 + SVD 刚体变换 + ICP，取点云重叠分最高者"]
            M3 --> M4["人体姿态估计得手/脚 2D 位置 → 变换到 3D 接触位置 L<br/>并投影到物体最近表面"]
        end
        GEN --> RT["3. 重定向 Retargeting<br/>WHAM 估计人体关节/足位/基座朝向 → IK 得到机器人构型 C"]
        CT --> RT
    end
    M4 --> KF["关键帧 Keyframe: 接触位置 L + 机器人构型 C"]
    RT --> KF
    KF --> TO
    subgraph TO["全身轨迹优化 TO §IV"]
        T1["质心动力学 + 全身运动学"] --> T2["接触约束 (面接触等) + 碰撞惩罚"]
        T2 --> T3["关键帧代价项引导基座位姿等"]
    end
    TO --> OUT["物理一致的长时程 loco-manipulation 轨迹<br/>(MuJoCo 中 Unitree G1 验证)"]
```

图依据：论文 §III（Fig.2、Fig.3、Fig.4）、§IV（式 (1)(9)(14)）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/Physically_Consistent_Humanoid_Loco-Manipulation_using_Latent_Diffusion_Models` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页当前提供的是「待读条目」而不是「论文结论」：它确定的只是这篇工作在 04_Loco-Manipulation_and_WBC 下的归属与待深读状态，真正的方法取舍要等笔记落地后才谈得上。**

- 页面上唯一确定的信息是归类与计划路径：分类 04_Loco-Manipulation_and_WBC、计划文件夹 `papers/04_Loco-Manipulation_and_WBC/Physically_Consistent_Humanoid_Loco-Manipulation_using_Latent_Diffusion_Models`，深读状态仍为「待撰写」。
- 适用边界：可用于按分类检索、查缺补漏和交叉链接，不可当作该论文机制或指标的依据——本页未给出任何机制描述与量化结果。
- 主要风险是被误当成已消化的笔记：本页「实验与评测」已明确 benchmark、消融与实机指标待补，引用前应回到 progress.json 与原论文。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_physically-consistent-humanoid-loco-manipulation.md](../../sources/papers/humanoid_pnb_physically-consistent-humanoid-loco-manipulation.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2504.16843)：已通读方法部分（实验与附录以略读为准）：§III 规划流程、§IV 轨迹优化主要公式、§V 实验设置；附录未读

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
