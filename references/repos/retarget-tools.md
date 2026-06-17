# Retarget Tools

**Retarget Tools：** 人体/动物运动到机器人执行空间的开源工具与代表性项目导航（几何重定向、物理补丁、视频/单目估计与轨迹编辑），与知识库 [Motion Retargeting（动作重定向）](../../wiki/concepts/motion-retargeting.md) 主线配套使用。

## 与知识库主线的关系

- 任务定义与分类坐标：[Motion Retargeting](../../wiki/concepts/motion-retargeting.md)
- 运动学前端基线：[GMR（通用动作重定向）](../../wiki/methods/motion-retargeting-gmr.md)
- 学习式整段映射 + 仿真修补：[NMR](../../wiki/methods/neural-motion-retargeting-nmr.md)
- 物理感知双层 RL 参考生成：[ReActor](../../wiki/methods/reactor-physics-aware-motion-retargeting.md)
- 下游模仿学习语境：[Imitation Learning](../../wiki/methods/imitation-learning.md)

## 人体表示与参数化模型

- [三维人体动捕模型 SMPL：A Skinned Multi Person Linear Model](https://yunyang1994.github.io/2021/08/21/%E4%B8%89%E7%BB%B4%E4%BA%BA%E4%BD%93%E6%A8%A1%E5%9E%8B-SMPL-A-Skinned-Multi-Person-Linear-Model/) — SMPL 系表示是多数重定向与视频人体估计管线的共同前置语言。
- [fairmotion](https://github.com/facebookresearch/fairmotion) — Meta 通用动捕数据处理库（BVH/AMASS IO、3D 变换、FK、可视化）；重定向上游数据基础设施，本身不做机器人重定向（2023 已归档）；wiki [fairmotion](../../wiki/entities/fairmotion.md)

## 人形：几何重定向与通用实现

- [GMR](https://github.com/YanjieZe/GMR) — 通用运动学前端；wiki [GMR](../../wiki/methods/motion-retargeting-gmr.md)
- [holosoma](https://github.com/amazon-far/holosoma) / [OmniRetarget](https://omniretarget.github.io/) — 交互保留重定向 + WBT 训练；wiki [holosoma](../../wiki/entities/holosoma.md)
- [mocap_retarget](https://github.com/ccrpRepo/mocap_retarget) — 工程向 MoCap 管线；wiki [mocap-retarget](../../wiki/entities/mocap-retarget.md)
- [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter) — SOMA BVH→G1 CSV；wiki [soma-retargeter](../../wiki/entities/soma-retargeter.md)

## 人形：物理一致性、策略学习与集成仿真

- [PHC](https://github.com/ZhengyiLuo/PHC) — SMPL fitting + 物理模仿；wiki [PHC](../../wiki/entities/phc.md)
- [ProtoMotions](https://github.com/NVlabs/ProtoMotions) — 大规模并行仿真；wiki [ProtoMotions](../../wiki/entities/protomotions.md)
- [MimicKit](https://github.com/xbpeng/MimicKit) — 轻量模仿学习 + GMR 转换；wiki [MimicKit](../../wiki/entities/mimickit.md)
- [SPIDER](https://github.com/facebookresearch/spider) — 物理采样式重定向；wiki [SPIDER](../../wiki/methods/spider-physics-informed-dexterous-retargeting.md)
- [sbto](https://github.com/Atarilab/sbto) / [DynaRetarget](https://atarilab.github.io/dynaretarget.io/) — 增量 SBTO 动力学 refinement（OmniRetarget 参考 → MuJoCo CEM）；wiki [sbto](../../wiki/entities/sbto.md)
- [human2humanoid](https://github.com/LeCAR-Lab/human2humanoid) — 遥操 + AMASS 重定向；wiki [human2humanoid](../../wiki/entities/human2humanoid.md)
- [AMP-RSL-RL](https://github.com/gbionics/amp-rsl-rl) — rsl_rl(PPO)+AMP 人形模仿，对称性增广、可 pip 安装（IIT）；wiki [amp-rsl-rl](../../wiki/entities/amp-rsl-rl.md)

## 人形：视频与单目人体运动

- [GVHMR](https://github.com/zju3dv/GVHMR) — 单目视频→SMPL；wiki [GVHMR](../../wiki/entities/gvhmr.md)
- [VideoMimic](https://github.com/hongsukchoi/VideoMimic) — 视频驱动人形模仿；wiki [VideoMimic](../../wiki/entities/videomimic.md)

## 四足：模仿动物与 AMP 生态

- [motion_imitation](https://github.com/erwincoumans/motion_imitation) — 奠基四足模仿动物；wiki [motion-imitation-quadruped](../../wiki/entities/motion-imitation-quadruped.md)
- [STMR 官方项目页](https://taerimyoon.me/Spatio-Temporal-Motion-Retargeting-for-Quadruped-Robots/) — 四足时空重定向（arXiv:2404.11557）；wiki [stmr-quadruped-retargeting](../../wiki/entities/stmr-quadruped-retargeting.md)（注：原 `terry97-guel/*` GitHub 子仓已 404）
- [AMP_for_hardware](https://github.com/escontra/AMP_for_hardware) — 四足 AMP 工程基座（Escontrela）；wiki [amp-for-hardware](../../wiki/entities/amp-for-hardware.md)
- [MetalHead](https://github.com/inspirai/MetalHead) — A1 AMP jump/recovery；wiki [MetalHead](../../wiki/entities/metalhead.md)
- [LeggedGym-Ex](https://github.com/lupinjia/LeggedGym-Ex) — legged_gym 多仿真器 + AMP/DeepMimic；wiki [leggedgym-ex](../../wiki/entities/leggedgym-ex.md)
- [Go2 motion-imitation](https://github.com/TSUITUENYUE/motion-imitation) — Go2 retarget + Genesis；wiki [go2-motion-imitation](../../wiki/entities/go2-motion-imitation.md)

## 跨形态（人↔四足）研究向

- [pan-motion-retargeting](https://github.com/hlcdyy/pan-motion-retargeting) — 学习式部位注意力重定向；wiki [pan-motion-retargeting](../../wiki/entities/pan-motion-retargeting.md)
- [walk-the-dog](https://github.com/PeizhuoLi/walk-the-dog) — 相位流形跨形态对齐；wiki [walk-the-dog](../../wiki/entities/walk-the-dog.md)

## 轨迹与关键帧编辑（机器人运动资产）

- [机器人关键帧编辑器](https://github.com/cyoahs/robot_motion_editor)
- [robot-keyframe-kit](https://github.com/Stanford-TML/robot_keyframe_kit)
- [Robot Motion Editor](https://github.com/project-instinct/robot-motion-editor)

## 本页参考

- 外链清单沿革自归档文件 [sources/retarget.md](../../sources/retarget.md)（旧版 README 资源地图），本页按用途分组并互链到 wiki 主线。
