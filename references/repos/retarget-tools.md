# Retarget Tools

**Retarget Tools：** 人体运动到机器人执行空间的开源工具与代表性项目导航（几何重定向、物理补丁、视频/单目估计与轨迹编辑），与知识库 [Motion Retargeting（动作重定向）](../../wiki/concepts/motion-retargeting.md) 主线配套使用。

## 与知识库主线的关系

- 任务定义与分类坐标：[Motion Retargeting](../../wiki/concepts/motion-retargeting.md)
- 运动学前端基线：[GMR（通用动作重定向）](../../wiki/methods/motion-retargeting-gmr.md)
- 学习式整段映射 + 仿真修补：[NMR](../../wiki/methods/neural-motion-retargeting-nmr.md)
- 物理感知双层 RL 参考生成：[ReActor](../../wiki/methods/reactor-physics-aware-motion-retargeting.md)
- 下游模仿学习语境：[Imitation Learning](../../wiki/methods/imitation-learning.md)

## 人体表示与参数化模型

- [三维人体动捕模型 SMPL：A Skinned Multi Person Linear Model](https://yunyang1994.github.io/2021/08/21/%E4%B8%89%E7%BB%B4%E4%BA%BA%E4%BD%93%E6%A8%A1%E5%9E%8B-SMPL-A-Skinned-Multi-Person-Linear-Model/) — SMPL 系表示是多数重定向与视频人体估计管线的共同前置语言。

## 几何重定向与通用实现

- [GMR](https://github.com/YanjieZe/GMR) — 通用运动学前端；与 wiki 方法页 [GMR](../../wiki/methods/motion-retargeting-gmr.md) 对应。
- [OmniRetarget](https://omniretarget.github.io/) — 跨形态重定向框架与项目主页。
- [mocap_retarget](https://github.com/ccrpRepo/mocap_retarget?tab=readme-ov-file) — 动捕数据向机器人运动空间的工程向管线示例。

## 物理一致性、策略学习与集成仿真

- [PHC](https://github.com/ZhengyiLuo/PHC) — 人形/角色控制与重定向相关 RL 生态中的常用基座之一。
- [ProtoMotions](https://github.com/NVlabs/ProtoMotions) — NVIDIA 侧大规模可微分运动与物理集成方向的代表仓库。
- [SPIDER](https://github.com/facebookresearch/spider) — Meta 研究向的物理交互与运动表示相关开源（与重定向/控制管线常并列出现）。

## 视频与单目人体运动

- [GVHMR](https://github.com/zju3dv/GVHMR?tab=readme-ov-file) — 视频/单目场景下的人体运动估计，可作为重定向上游观测。
- [VideoMimic](https://github.com/hongsukchoi/VideoMimic) — 视频驱动模仿与运动先验相关仓库。

## 轨迹与关键帧编辑（机器人运动资产）

- [机器人关键帧编辑器](https://github.com/cyoahs/robot_motion_editor)
- [robot-keyframe-kit](https://github.com/Stanford-TML/robot_keyframe_kit)
- [Robot Motion Editor](https://github.com/project-instinct/robot-motion-editor)

## 本页参考

- 外链清单沿革自归档文件 [sources/retarget.md](../../sources/retarget.md)（旧版 README 资源地图），本页按用途分组并互链到 wiki 主线。
