---
type: entity
project_id: robopace
tags: [paper, manipulation, bimanual, vla, action-chunking, retiming, contact-aware, pi05, openarm]
status: complete
updated: 2026-10-12
arxiv: "2610.09696"
project: https://robopace.airoa.io/
venue: "arXiv 2026"
summary: "RoboPace：不重训策略、保持动作块几何路径，用 TOPP-RA 在线重定时并在预测接触处限速；OpenArm 双臂三类接触密集任务上 32/50 成功，优于 10 Hz 均匀减速的 22/50，且耗时约为其 45-59%。"
related:
  - ../overview/frontier-manipulation-2026-10-05-10-09.md
  - ../tasks/bimanual-manipulation.md
  - ../tasks/manipulation.md
  - ../methods/action-chunking.md
  - ./paper-pi05-open-world-vla.md
  - ./paper-umi-ft.md
sources:
  - ../../sources/papers/robopace_arxiv_2610_09696.md
  - ../../sources/sites/robopace-project.md
  - ../../sources/blogs/frontier_manipulation_2026_10_05_10_09.md
---

# RoboPace

**RoboPace**（arXiv:2610.09696，Shirasaka / Ohkawa / … / Ota）是一个**在线接触感知重定时层**：保持 VLA 输出动作块的几何路径不变，只改变时间参数化——自由空间尽量按机器人运动学/动力学极限加速，预测到接触时限速，无需重训，实时运行。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TOPP-RA | Time-Optimal Path Parameterization based on Reachability Analysis | 给定路径与速度/加速度约束求时间最优参数化 |
| UMI | Universal Manipulation Interface | 手持夹爪式无机器人示范采集 |
| VLA | Vision-Language-Action model | 视觉语言动作模型，本文基于 π0.5 |
| stc | steps-to-contact | 学习线索预测的距接触源步数 |
| DLS-IK | Damped Least-Squares IK | 阻尼最小二乘逆运动学，把笛卡尔 chunk 转关节路径 |
| RACE | retiming without contact gate | 论文中“仅物理极限重定时”的基线 |

## 为什么重要

- 人类示范的时序被 VLA 继承，但人手柔顺、机器人不柔顺：均匀加速易过冲（50 Hz 仅 4/50），均匀减速又太慢。
- 重定时层与策略解耦，可作为任意 action-chunk 策略的外挂。

## 方法栈与流程总览

```mermaid
flowchart LR
  P["π0.5 策略: 动作块 (32 步)"] --> IK["DLS-IK 转关节路径"]
  IK --> Q["接触线索: 几何距离 + EfficientNet-B0/GRU 的 stc"]
  Q --> V["接触路点: 笛卡尔手速上限 v_contact (余弦斜坡 rho=4)"]
  V --> T["TOPP-RA: 关节速度/加速度约束下时间最优"]
  T --> B["新 chunk 与在飞计划线性/余弦/tanh 混合"]
  B --> R["OpenArm 执行; 每 16 源步重查策略"]
```

## 方法

- **重定时**：TOPP-RA 逐 chunk 求解，约束为关节速度/加速度上限，加上接触关键路点上的笛卡尔手速上限（转为沿路径的关节速度上界）。
- **接触线索**：几何线索（双手距离 < 0.08 m 或距桌面 < 0.02 m）；学习线索为 EfficientNet-B0 + GRU（4.69M 参数），由腕部相机与夹爪角度预测 stc，≤ 20 源步（扣延迟）即标记，训练用 8,601 条录制按录制划分，起始预测误差约 0.80 源步（0.027 s）。
- **在线运行**：每执行 16 个源步重查策略，新 chunk 以 Linear/Cosine/Tanh 调度并入；每 chunk 计算耗时论文未报告。
- **策略与平台**：π0.5 微调（32 步视野，双腕相机），用 8,434 小时 YUBI 录制预训练；双臂 7-DoF OpenArm v1.0 + YUBI 手，RTX 5070 Ti。数据 Cup/Socks 8,736 条、Pyramid 5,288 条人工采集 episode。

## 实验与评测

- 3 个任务 5 个指令 × 10 次 = 50 次。成功：RoboPace(Linear-N3) 32/50；Decelerated(10 Hz) 22/50；Accelerated(50 Hz) 4/50；RACE 式 1/50；Demo 速度(30 Hz) 总数未给，仅称 RoboPace 在四个指令上更快、Socks 更慢。
- 在 Decelerated 的成功试验中，RoboPace 仅用 45–59% 的时间（四个指令）。
- v_contact 敏感性（Pyramid 阶段 2）：0.01 / 0.02 / 0.04 m/s，成功率随上限升高而下降，选 0.02（Socks 用 0.04）。
- 过渡调度：Linear-N3 32/50，Tanh-N3 27/50，Tanh-N9 25/50。
- 交接（Part，单次）：299 源步 48 s，接触门控下速度中位数 0.01 m/s。

## 源码运行时序图

不适用：论文未声明代码或权重发布，项目页（robopace.airoa.io）在 2026-10-12 域名解析失败，未能核实；无官方入口可对齐。

## 工程实践

- 接入点在“策略输出之后、控制器之前”：策略 → IK → 重定时 → 关节指令。
- v_contact 需按任务调：Socks 与 Pyramid 取值不同。
- 开源状态：**未列链接 / 项目页未核实**（2026-10-12）。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| RACE | 同为 TOPP 重定时但无接触门控；本文消融中仅 1/50 |
| SAIL | 切换预设速度；论文仅在相关工作中讨论，未实验对比 |
| [π0.5](./paper-pi05-open-world-vla.md) | RoboPace 的策略底座，不改其权重 |
| [Action Chunking](../methods/action-chunking.md) | 本文处理 chunk 的时间维而非内容 |

## 结论

**RoboPace 的价值是在不碰策略的前提下，用接触门控让“快”和“稳”并存；真影响指标是成功率与耗时之比。**

1. **均匀加速不可行** — 50 Hz 只有 4/50。
2. **接触限速是关键** — 去掉接触门控的 RACE 式只有 1/50。
3. **v_contact 是主旋钮** — 越高越差，需按任务调。
4. **耗时收益有条件** — 对 10 Hz 基线省 41–55%，但 Socks 比示范速度还慢。
5. **样本小** — 50 次试验、单机器人、单策略族。

## 局限与风险

- 只在一台机器人、一类策略上评测；几何线索与相机线索未单独消融。
- 最终放置前相机线索不减速，几何线索不建模被抬起的容器。
- 项目页未能核实，代码状态未知。

## 关联页面

- [Manipulation 周报汇总（2026-10-05 至 10-09）](../overview/frontier-manipulation-2026-10-05-10-09.md)
- [双臂操作任务页](../tasks/bimanual-manipulation.md)
- [Action Chunking 方法页](../methods/action-chunking.md)
- [Manipulation 任务页](../tasks/manipulation.md)

## 参考来源

- [robopace_arxiv_2610_09696.md](../../sources/papers/robopace_arxiv_2610_09696.md)
- [robopace-project.md](../../sources/sites/robopace-project.md) — 项目页核查（抓取失败）
- [frontier_manipulation_2026_10_05_10_09.md](../../sources/blogs/frontier_manipulation_2026_10_05_10_09.md) — 周报线索
- 论文：<https://arxiv.org/abs/2610.09696>

## 推荐继续阅读

- [TOPP-RA（Pham & Pham）](https://arxiv.org/abs/1707.07239)
- [Universal Manipulation Interface](https://umi-gripper.github.io/)
