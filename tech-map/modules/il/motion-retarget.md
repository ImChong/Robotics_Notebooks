---
type: tech_map_node
status: complete
topic: [motion-imitation, imitation-learning]
summary: "Motion Retargeting（动作重定向）技术栈模块：将人体/动物动作映射为满足机器人骨骼比例、关节限位与接触约束的可执行参考轨迹。"
sources:
  - ../../../sources/notes/retarget.md
  - ../../../sources/papers/modern_robotics_textbook.md
updated: 2026-11-24
---

# Motion Retargeting（动作重定向 · 技术栈模块）

人的动作数据不能直接喂给机器人：骨骼长度、关节数量、力矩上限、可行接触都不一样。**动作重定向** 就是把人体动作映射成机器人可执行的参考轨迹，它是「人类动作数据 → 人形技能」这条数据链上绕不开的一环，质量直接决定后续跟踪与模仿的上限。

## 在全栈中的位置

```mermaid
flowchart LR
    RAW["人类动作源：MoCap / 单目视频 / VR 遥操"] --> SMPL["人体骨架/SMPL 参数化"]
    SMPL --> KIN["几何/运动学重定向（IK + 比例缩放）"]
    KIN --> PHYS["动力学与接触可行性修正（轨迹优化 / 仿真）"]
    PHYS --> POL["下游全身跟踪 RL / 模仿学习（AMP / WBT）"]
```

## 核心内容与站内入口

- **概念与流水线**：[Motion Retargeting](../../../wiki/concepts/motion-retargeting.md) · [重定向流水线](../../../wiki/concepts/motion-retargeting-pipeline.md)
- **技术路线对比与纵深**：[三条技术路线对照](../../../wiki/queries/motion-retargeting-three-routes-landscape.md) · [纵深路线](../../../roadmap/depth-motion-retargeting.md)
- **开源工具索引**：[Retarget Tools](../../../references/repos/retarget-tools.md)

## 上下游模块

- **上游基础**：[运动学](../robotics/kinematics.md) · [刚体运动](../robotics/rigid-body-motion.md) · [全栈技术域总览](../../overview.md)
- **下游消费者**：[人形强化学习](../rl/humanoid-rl.md) · [Behavior Cloning](behavior-cloning.md)

## 参考来源与收录边界

- [sources/notes/retarget.md](../../../sources/notes/retarget.md) — 重定向工具生态归档
- [Modern Robotics 教材摘录](../../../sources/papers/modern_robotics_textbook.md) — 运动学与雅可比数学基础
- 各开源重定向工具的具体依赖见 [Retarget Tools](../../../references/repos/retarget-tools.md)。
