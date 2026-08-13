---
type: entity
tags:
  - paper
  - whole-body-planning
  - humanoid
  - confined-space
  - trajectory-optimization
  - residual-rl
  - unitree
  - ut-austin
status: complete
updated: 2026-08-13
arxiv: "2608.10220"
related:
  - ../concepts/whole-body-control.md
  - ../tasks/humanoid-locomotion.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../concepts/whole-body-tracking-pipeline.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/g1_confined_space_wbp_arxiv_2608_10220.md
  - ../../sources/sites/confined-space-wbp-humanoid-github-io.md
summary: "G1 Confined-Space WBP（arXiv:2608.10220，UT Austin）：三阶段全身规划（环境 TO→可微 SCA→全阶动力学）+ 残差 RL 跟踪；Unitree G1 穿越超 NIST 狭窄环境；截至入库日代码未开源。"
---

# G1 Confined-Space WBP（狭窄空间全身规划 · arXiv:2608.10220）

**G1 Confined-Space WBP**（*Whole-Body Planning for Humanoids Navigating Confined Spaces via Self-Collision Avoidance References*，[arXiv:2608.10220](https://arxiv.org/abs/2608.10220)）由 **德州大学奥斯汀分校（UT Austin）** Carlos Gonzalez / Luis Sentis 提出：在接触序列给定时，用 **刚体体积可微自碰引导** 把全阶轨迹优化拉出狭窄非凸局部最小，再以残差 RL 在线跟踪。[项目页](https://carlosiglezb.github.io/confined-space-wbp-humanoid/)。

## 一句话定义

**别用粒子样条硬闯窄口：先在可达刚体体积上做可微 SCA 引导，再解动力学可行全身轨迹，最后用残差策略跟住计划。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBP | Whole-Body Planning | 全身轨迹规划（相对瞬时 WBC） |
| SCA | Self-Collision Avoidance | 自碰规避；Stage2/3 核心约束 |
| TO | Trajectory Optimization | 轨迹优化；Stage1/3 求解器 |
| \(C_r\) | Confinement Ratio | \(E_{\mathrm{ca}}/A_{\mathrm{ca}}\)；\(C_r<2\) 表受限机动 |
| NIST | National Institute of Standards and Technology | 应急响应狭窄环境标准参照 |

## 为什么重要

- **狭窄空间是人形落地痛点：** 构型流形窄、非凸，ROM/样条引导常在 Hole 类场景 **0/10**。
- **规划与学习分工清楚：** 优化出长视界（12–18 s）多接触参考，RL 只学残差跟踪 + DR。
- **可量化难度：** 用 \(C_r\) 与 NIST 对照，便于和其他 confined locomotion 工作对齐。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 德州大学奥斯汀分校（UT Austin） |
| **平台** | Unitree G1（27-DoF） |
| **开源** | **截至 2026-08-13 未开源**（项目页无 Code；Paper 按钮占位） |
| **源码运行时序图** | **不适用**（无可运行官方实现） |

## 核心原理

### 三阶段管线

| 阶段 | 做什么 |
|------|--------|
| Stage 1 | 环境感知任务空间 TO：Bézier 控制躯干/手足路径，IRIS 区域 + 可达性（含膝） |
| Stage 2 | 胶囊/球原语可微碰撞（SOCP \(\alpha\geq 1\)） refinement，修正自碰 |
| Stage 3 | 全阶动力学 WBP：跟踪 Stage2 引导，硬碰撞/关节/力矩约束，摩擦锥 soft barrier；两遍（先动力学后硬碰撞） |

### 闭环跟踪

非对称 actor-critic 残差 PPO：策略吃噪声本体 + 跟踪误差 + 局部障碍距离 + 接触日程 look-ahead；输出 \(\delta a\) 叠加参考关节目标；DR 含质量/摩擦/推扰。

### 流程总览

```mermaid
flowchart LR
  CS[接触序列]
  S1[Stage1 环境 TO]
  S2[Stage2 可微 SCA]
  S3[Stage3 全阶 WBP]
  PI[残差 RL 跟踪]
  CS --> S1 --> S2 --> S3 --> PI
```

## 工程实践

| 项 | 建议读法 |
|----|----------|
| 假设 | **接触序列外给**；本工作不解决接触搜索 |
| 求解成本 | 分钟级（约 2–6 min/试验，CPU+消费级 GPU） |
| 碰撞模型 | Stage2 粗原语引导；Stage3 更细关节级几何硬约束 |
| 控制器 | Unitree RL mjlab + PPO；同一结构跨三环境 |
| 开源跟进 | 盯项目页 Code / 论文相机就绪声明 |

## 实验与评测

三测试床：Tilted Stairs、Unobstructed Hole、Obstructed Hole（超 NIST 应急标准；Hole \(C_r\approx 1.4\)–\(1.5\)）。

| 配置 | Stairs | Unobstr. Hole | Obstr. L/R |
|------|--------|---------------|------------|
| Full Stage1+2+3 | **10/10** | **10/10** | **7/10 / 6/10** |
| 无膝样条基线 | 9/10 | 0/10 | 0/10 |
| w/o Stage2 | 10/10 | 10/10 | 1/10 / 0/10 |

残差策略在完整 DR 下遍历成功率 **>95%**（仿真）。

## 结论

**在 \(C_r<1.5\) 的窄口，成败关键是「体积感知 SCA 引导」能不能把全阶求解器放进可行盆；残差 RL 负责把分钟级计划变成可在线执行的跟踪。**

1. **Hole 环境先看有无膝引导 + Stage2** — 无膝样条与去 Stage2 都会在障碍孔崩溃。
2. **接触序列仍是上游问题** — 本页解决的是连续 WBP，不是混合整数接触搜索。
3. **真机未验证** — 选型时按「强仿真规划基准」读，不要当已部署栈。
4. **与 WBC 瞬时控制互补** — 长视界多接触计划 → 跟踪策略 → 底层 PD/WBC。

## 局限与风险

- 代码与硬件实验截至入库日未公开。
- 允许厘米级规划穿透，依赖控制器补偿。
- 求解时间与 Right-Step 方差大，难直接进高频重规划。

## 与其他工作对比

| 工作 | 关系 |
|------|------|
| Gonzalez & Sentis 2024 凸松弛引导 | Stage1 前身；本文升到刚体体积 SCA |
| ROM / 无膝样条 WBP | 在 Hole 上失败；本文对照基线 |
| 纯端到端 RL 狭窄导航 | 样本效率与死区问题；本文用计划作参考 |
| 模仿学习全身技能 | 狭窄姿态演示稀缺；本文走优化合成 |

## 关联页面

- [Whole-Body Control](../concepts/whole-body-control.md)
- [人形 locomotion](../tasks/humanoid-locomotion.md)
- [Whole-body tracking pipeline](../concepts/whole-body-tracking-pipeline.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [论文归档](../../sources/papers/g1_confined_space_wbp_arxiv_2608_10220.md)
- [项目页归档](../../sources/sites/confined-space-wbp-humanoid-github-io.md)

## 推荐继续阅读

- 项目页环境演示：<https://carlosiglezb.github.io/confined-space-wbp-humanoid/>
- 论文 HTML：<https://arxiv.org/html/2608.10220>
