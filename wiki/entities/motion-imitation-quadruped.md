---
type: entity
tags: [repo, quadruped, motion-imitation, motion-retargeting, deepmimic]
status: complete
updated: 2026-06-08
summary: "erwincoumans/motion_imitation 是 Peng 等「模仿动物」四足论文官方实现：动物 MoCap → 四足参考 → PyBullet RL 模仿，为 legged_gym/AMP 生态奠基。"
related:
  - ../tasks/locomotion.md
  - ../methods/deepmimic.md
  - ../methods/amp-reward.md
  - ./legged-gym.md
  - ./xue-bin-peng.md
sources:
  - ../../sources/repos/motion_imitation_peng.md
---

# motion_imitation（四足模仿动物）

**motion_imitation**（<https://github.com/erwincoumans/motion_imitation>）是 Xue Bin Peng 等论文 [*Learning Agile Robotic Locomotion Skills by Imitating Animals*](https://xbpeng.github.io/projects/Robotic_Imitation/index.html) 的 **官方开源实现**：在 **PyBullet** 中训练四足机器人跟踪 **动物 MoCap 参考**，是把「动物动作重定向到腿式机器人」带入主流 RL 社区的 **早期标杆仓库**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 示例引导模仿学习 |
| MoCap | Motion Capture | 动物参考动作来源 |
| AMP | Adversarial Motion Prior | 后续四足风格先验扩展 |
| PD | Proportional–Derivative | 仿真中关节跟踪底层 |

## 为什么重要

- **四足重定向的历史锚点**：虽不叫 retargeting 模块，但核心问题就是 **异构骨架（动物→四足）参考映射 + 可跟踪性**。
- **后续生态母本**：[DeepMimic](../methods/deepmimic.md)、[AMP](../methods/amp-reward.md)、[legged_gym](./legged-gym.md)、[MimicKit](./mimickit.md) 均由此脉络演化。

## 使用要点

- `paper` 分支保留论文原始代码；`motion_imitation/data/motions/` 含多种动物片段（pace、trot 等）。
- 训练示例：`python motion_imitation/run.py --mode train --motion_file motion_imitation/data/motions/dog_pace.txt --visualize`

## 流程概念

```mermaid
flowchart LR
  animal["动物 MoCap 片段"] --> map["骨架/关键点映射\n（隐式于环境）"]
  map --> ref["四足参考状态序列"]
  ref --> rl["PyBullet 模仿 RL"]
  rl --> policy["四足策略"]
```

## 关联页面

- [DeepMimic](../methods/deepmimic.md)
- [AMP 奖励设计](../methods/amp-reward.md)
- [legged_gym](./legged-gym.md)
- [Xue Bin Peng](./xue-bin-peng.md)

## 参考来源

- [motion_imitation 仓库归档](../../sources/repos/motion_imitation_peng.md)

## 推荐继续阅读

- 项目页：<https://xbpeng.github.io/projects/Robotic_Imitation/index.html>
- GitHub：<https://github.com/erwincoumans/motion_imitation>
