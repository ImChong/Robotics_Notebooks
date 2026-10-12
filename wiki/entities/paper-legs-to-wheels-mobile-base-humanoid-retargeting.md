---

type: entity
tags: [paper, humanoid, motion-retargeting, wheeled-humanoid, mobile-manipulation, robotis, gachon]
status: complete
updated: 2026-10-12
arxiv: "2610.08381"
project_id: legs-to-wheels-mobile-base-humanoid-retargeting
related:
  - ../overview/frontier-humanoid-quadruped-2026-10-05-10-09.md
  - ../concepts/motion-retargeting.md
  - ../methods/motion-retargeting-gmr.md
  - ./robotis-ai-worker.md
  - ./gvhmr.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/legs_to_wheels_arxiv_2610_08381.md
  - ../../sources/blogs/frontier_humanoid_quadruped_2026_10_05_10_09.md
summary: "From Legs to Wheels（arXiv:2610.08381）：嘉泉大学为 ROBOTIS AI Worker（轮式底盘+升降+双臂）提出任务条件的人体动作分配器，把步行/骨盆运动重分配给底盘、升降与手臂；9 场景归一化任务代价 0.322（Arm-only 0.759、Base-only 0.640），仿真回放 9/9 完成；实机仅定性，未发现公开代码。"
---

# From Legs to Wheels：面向轮式人形的形态感知人体动作重定向

**From Legs to Wheels**（[arXiv:2610.08381](https://arxiv.org/abs/2610.08381)）针对「没有腿」的移动底座人形（全向轮底盘 + 垂直升降 + 双臂），提出一个任务条件的动作分配与修复框架：先决定人体动作中哪部分交给底盘、升降轴或手臂，再映射到具体机器人并生成 50 Hz 部署指令。

## 一句话定义

**不是把人体关节逐个映射，而是按任务线索把人体的行走、升降与手部动作拆分后重分配给底盘、升降和手臂，同时保住手部任务空间轨迹。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GMR | General Motion Retargeting | 本文用于上半身重定向的基础方法 |
| IK / QP | Inverse Kinematics / Quadratic Programming | 在关节位置、速度限制下求关节速度 |
| SMPL | Skinned Multi-Person Linear model | 人体动作表示，来自 GVHMR 式单目重建 |
| DoF | Degrees of Freedom | AI Worker 单臂 7 DoF，参考共 31 关节 |
| GVHMR | Gravity-View Human Motion Recovery | 单目人体动作重建来源 |

## 为什么重要

- 轮式人形没有腿关节，腿式重定向范式（骨盆+腿跟踪）不适用，腰部升降与底盘速度需要新的分配规则。
- 单目重建的根高漂移、转弯时的多余平移是常见脏数据，需要在重定向阶段修复而非留给下游。
- 工作与 [ROBOTIS AI Worker](./robotis-ai-worker.md) 这类商用轮式人形平台直接对应。

## 核心信息

| 项 | 内容 |
|----|------|
| 机构 | 嘉泉大学（Gachon University）iRASC Lab |
| 平台 | ROBOTIS AI Worker（FFW-SG2），31 关节参考 |
| 输入 | 预先计算的 SMPL 风格动作（GVHMR 式） |
| 开源 | **未发现**项目页/代码（截至 2026-10-12） |

## 流程总览

```mermaid
flowchart LR
  A["单目视频 -> SMPL 动作 (GVHMR 式)"] --> B["任务标签: 行走/够取/弯腰/拾取"]
  B --> C["任务条件分配器: 底盘 / 升降 / 手臂 权重与目标"]
  C --> D["GMR 上半身重定向 + 差分 IK/QP"]
  D --> E["朝向稳定 + 速度重定时 + 升降/手臂修复"]
  E --> F["部署适配器: 50Hz, 平滑/死区/斜率限制"]
  F --> G["AI Worker 底盘速度 + 升降 + 双臂指令"]
```

## 方法

- **分配：** 行走线索驱动底盘并压低摆臂权重；手高/躯干俯仰/肩到手位移驱动升降；手腕轨迹与朝向驱动手臂并提高手部增益。输出阶段标签、底盘/升降/手臂权重、底盘目标轨迹、升降参考与手部目标。
- **朝向稳定：** 最小加加速度混合重建底盘朝向，转弯主导时抑制多余平移，并把移除的平移重分配到非转弯区间保净位移。
- **重定时：** s=max(1,max‖v‖/v_max,max|ω|/ω_max)，输出频率 f_out=f_in/s，只改时长不改路径。
- **升降/手臂修复：** 升降门控拒绝单目高度漂移、保留有意下蹲；手臂过渡用稳定起点最小加加速度轨迹；低物体拾取用「开-合」两阶段与升降下降协调。
- **部署适配：** 50 Hz；抖动判定（路径/净位移比高而净位移小则保持静止）；滑动平均、线/角死区、加速度斜率限制；增益 g_lin、g_ang 可标定（数值原文未报告）。

## 实验与评测

| 方法 | 行走 01–04 | 操作 05–09 | 全部 01–09 |
|------|-----------|-----------|-----------|
| Ours | 0.355 | 0.295 | 0.322 |
| Arm-only | 1.000 | 0.566 | 0.759 |
| Base-only | 0.355 | 0.868 | 0.640 |
| No-lift | 0.355 | 0.440 | 0.402 |

（归一化任务空间代价，越低越好。）

- **分项：** 行走根误差 Ours/Base-only/No-lift 均 0.641 m，Arm-only 1.610 m；手腕误差 Ours 0.346 m，Base-only 1.726 m；升降缺口 Ours 0.012 m，其余 0.125 m。
- **修复前后：** 宽臂样本峰值关节速度 4.789→2.201 rad/s，峰值加速度 28.128→3.755 rad/s²；低拾取双手间距约 0.89→0.52 m，升降降到 −0.18 m。
- **仿真回放（无策略）：** 9/9 完成，body mean 0.020 m，anchor mean 0.097 m，2,947 步。
- **实机：** 仅场景 06，升降与双臂协调保留，定性，无成功率。

## 源码运行时序图

不适用：论文与 arXiv 页面均未提供代码仓库或项目页，截至 2026-10-12 无可运行入口。

## 工程实践

| 项 | 建议 |
|----|------|
| 轮式人形重定向 | 先分配（底盘/升降/手臂），再做上半身 IK，而不是直接套腿式 GMR |
| 单目输入 | 对根高度漂移加门控；转弯时分离平移与朝向 |
| 指令平滑 | 50 Hz 下加死区、滑动平均与加速度斜率限制，并保持二次校验 |
| 复现 | 无官方代码；增益、死区数值需自行标定 |

## 结论

**对轮式人形，先按任务重分配「谁来动」比提升末端逐点精度更重要；本文给出可解释的规则式方案，但评测规模小。**

1. **升降轴是关键：** 无升降的 No-lift 操作代价 0.440，高于 Ours 的 0.295。
2. **仅用手臂或仅用底盘都失败：** Arm-only 行走代价 1.000；Base-only 手腕误差 1.726 m。
3. **修复步骤明显降动力学峰值：** 加速度峰值降约 7.5 倍。
4. **读数留意规模：** 9 个场景、预计算动作、归一化代价。
5. **实机仅定性：** 完成率与误差未系统测量。
6. **无开源：** 参数需自行标定。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [GMR 方法页](../methods/motion-retargeting-gmr.md) | 腿式/人体近似重定向基础；本文在其上增加形态感知分配（上半身仍由 GMR 完成） |
| [运动重定向概念](../concepts/motion-retargeting.md) | 本文属于异构形态（轮式）分支 |
| [GVHMR](./gvhmr.md) | 本文的单目动作重建来源类型 |

## 局限与风险

- **开源状态：** 截至 2026-10-12 未发现项目页、GitHub 或权重。
- 预计算人体动作；分配器含场景条件，缺少直接 GMR 无分配基线与留出评测。
- 回放不检验扰动抑制；实机定性；违规率与平滑度无数值。

## 关联页面

- [人形/四足前沿周报 2026-10-05~10-09](../overview/frontier-humanoid-quadruped-2026-10-05-10-09.md)
- [运动重定向概念](../concepts/motion-retargeting.md)
- [GMR 重定向](../methods/motion-retargeting-gmr.md)
- [ROBOTIS AI Worker](./robotis-ai-worker.md)

## 参考来源

- [legs_to_wheels_arxiv_2610_08381.md](../../sources/papers/legs_to_wheels_arxiv_2610_08381.md) — 论文来源归档
- [人形/四足周报来源](../../sources/blogs/frontier_humanoid_quadruped_2026_10_05_10_09.md)
- [arXiv:2610.08381](https://arxiv.org/abs/2610.08381)

## 推荐继续阅读

- [arXiv HTML 全文](https://arxiv.org/html/2610.08381v1)
- [GMR 论文](https://arxiv.org/abs/2510.02252)
