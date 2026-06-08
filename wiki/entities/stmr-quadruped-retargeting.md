---
type: entity
tags: [repo, quadruped, motion-retargeting, spatio-temporal, legged-gym]
status: complete
updated: 2026-06-08
summary: "STMR 四足时空重定向生态：Quadruped_Retargeting（SMR）+ Quadruped-Motion-Timing（TMR）+ STMR_RL 完整训练，将噪声动物关键点转为 Go1/A1 等可跟踪全身运动。"
related:
  - ../concepts/motion-retargeting.md
  - ./legged-gym.md
  - ../tasks/locomotion.md
  - ./motion-imitation-quadruped.md
sources:
  - ../../sources/repos/stmr_quadruped_retargeting.md
---

# STMR 四足时空重定向

**STMR**（*Spatio-Temporal Motion Retargeting for Quadruped Robots*，arXiv:[2404.11557](https://arxiv.org/abs/2404.11557)）把 **动物/视频关键点轨迹** 转为 **四足机器人全身可执行参考**，显式拆分 **空间重定向（SMR）** 与 **时间重定向（TMR）**，再通过 [STMR_RL](https://github.com/terry97-guel/STMR_RL) 接入 **legged_gym** RL 训练与真机部署。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| STMR | Spatio-Temporal Motion Retargeting | 空间+时间联合重定向 |
| SMR | Spatial Motion Retargeting | 关键点→机器人空间位形 |
| TMR | Temporal Motion Retargeting | 相位/时间轴对齐与基座运动补全 |
| RL | Reinforcement Learning | 跟踪重定向参考 |

## 为什么重要

- **四足专用重定向栈**：相比只改关节角的早期模仿，STMR 针对 **物理不可行参考** 与 **缺失基座运动** 两类痛点分阶段处理。
- **支持噪声源**：论文展示手持视频、野外动物轨迹等来源，经 STMR 后可训练策略并完成硬件实验（含跑酷类技能）。

## 三仓库分工

| 仓库 | 链接 | 职责 |
|------|------|------|
| Quadruped_Retargeting | https://github.com/terry97-guel/Quadruped_Retargeting | SMR 空间重定向 |
| Quadruped-Motion-Timing | https://github.com/terry97-guel/Quadruped-Motion-Timing | TMR 时间重定向 |
| STMR_RL | https://github.com/terry97-guel/STMR_RL | legged_gym 训练/播放/导出 |

训练示例：`python legged_gym/scripts/train.py --task={ROBOT}_{MR}_{MOTION}`（`ROBOT`∈go1,a1,al；`MR`∈NMR,SMR,TMR,STMR 等）

## 流程总览

```mermaid
flowchart LR
  kp["动物/视频关键点"] --> smr["SMR\n空间重定向"]
  smr --> tmr["TMR\n时间/基座补全"]
  tmr --> ref["四足全身参考"]
  ref --> rl["STMR_RL + legged_gym"]
  rl --> hw["Go1 / A1 真机"]
```

## 关联页面

- [Motion Retargeting](../concepts/motion-retargeting.md)
- [motion_imitation（四足）](./motion-imitation-quadruped.md)
- [legged_gym](./legged-gym.md)
- [Locomotion](../tasks/locomotion.md)

## 参考来源

- [STMR 生态仓库归档](../../sources/repos/stmr_quadruped_retargeting.md)

## 推荐继续阅读

- 论文 HTML：<https://arxiv.org/html/2404.11557v3>
- STMR_RL：<https://github.com/terry97-guel/STMR_RL>
