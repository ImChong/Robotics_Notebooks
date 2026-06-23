---

type: entity
tags: [repo, quadruped, motion-retargeting, spatio-temporal, legged-gym, paper, eth]
status: complete
updated: 2026-06-08
summary: "STMR 四足时空重定向：论文官方项目页 + arXiv；SMR/TMR 分解将噪声动物关键点转为可跟踪全身参考；原 README 引用的 terry97-guel GitHub 子仓截至 2026-06 已不可公开访问。"
related:
  - ../concepts/motion-retargeting.md
  - ./legged-gym.md
  - ../tasks/locomotion.md
  - ./motion-imitation-quadruped.md
sources:
  - ../../sources/repos/stmr_quadruped_retargeting.md
---

# STMR 四足时空重定向

**STMR**（*Spatio-Temporal Motion Retargeting for Quadruped Robots*，IEEE T-RO 2025，arXiv:[2404.11557](https://arxiv.org/abs/2404.11557)）把 **动物/视频关键点轨迹** 转为 **四足机器人全身可执行参考**，显式拆分 **空间重定向（SMR）** 与 **时间重定向（TMR）**，再接 **legged_gym** 式 RL 模仿学习与真机部署。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| STMR | Spatio-Temporal Motion Retargeting | 空间+时间联合重定向 |
| SMR | Spatial Motion Retargeting | 关键点→机器人空间位形 |
| TMR | Temporal Motion Retargeting | 相位/时间轴与基座运动补全 |
| RL | Reinforcement Learning | 跟踪重定向参考 |

## 为什么重要

- **四足专用重定向栈**：针对 **物理不可行参考** 与 **缺失基座运动** 分阶段处理；论文含 Go1 / Aliengo / B2 等真机视频。
- **支持噪声源**：手持视频等来源经 STMR 后可训练 RL 跟踪策略（含跳跃、后空翻等飞行相技能）。

## 一手资料入口（已验证可打开）

| 类型 | 链接 | 说明 |
|------|------|------|
| **论文** | <https://arxiv.org/abs/2404.11557> | 一手方法描述；正文声明代码见下方项目页 |
| **官方项目页** | <https://taerimyoon.me/Spatio-Temporal-Motion-Retargeting-for-Quadruped-Robots/> | 作者 Taerim Yoon 维护；含真机实验视频与方法说明 |
| **旧 gh-pages 跳转** | <https://terry97-guel.github.io/STMR-RL.github.io/> | 301 跳转至上述官方项目页 |

> **链接核验说明（2026-06-08）：** 论文与部分二手 README 曾引用 `github.com/terry97-guel/STMR_RL`、`Quadruped_Retargeting`、`Quadruped-Motion-Timing`，经 HTTP/GitHub API 核验均为 **404 / Not Found**，且不在 `terry97-guel` 当前公开仓库列表中。本页**不以失效 GitHub URL 作为一手入口**；获取代码请以 **arXiv 论文 + 官方项目页** 或联系作者为准。

## 流程总览

```mermaid
flowchart LR
  kp["动物/视频关键点"] --> smr["SMR\n空间重定向"]
  smr --> tmr["TMR\n时间/基座补全"]
  tmr --> ref["四足全身参考"]
  ref --> rl["legged_gym + RL"]
  rl --> hw["Go1 / Aliengo / B2 等真机"]
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
- 官方项目页：<https://taerimyoon.me/Spatio-Temporal-Motion-Retargeting-for-Quadruped-Robots/>
