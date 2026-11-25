# 2026 年人形机器人全身控制：12 个开源项目（深蓝具身智能，2026-11-23）

> 来源归档（用户提供完整文章的结构化摘要；尊重原文版权，不重复转载全文）

- **标题：** 2026年人形机器人全身控制（WBC），“卷”到哪一步了？
- **作者：** 深蓝具身智能；编辑：阿豹；审编：具身君
- **发布时间：** 2026-11-23 11:23（用户给定）
- **原文：** <https://mp.weixin.qq.com/s/8qS01YkvZ2-lvSo0cf4gSQ>
- **类型：** blog / 2026 开源项目综述
- **核对范围：** 用户提供全文中的三条研究主线、12 个项目及其官方 GitHub 链接、总结与价值判断；没有将本文二手结论直接冒充各论文实验结果。
- **通读：** 已完成

## 核心归纳

1. **规模化全身跟踪：** HoloMotion、SONIC / GR00T-WBC、OpenTrack、BeyondMimic、SuperMimic。把人体参考动作的重定向、RL 跟踪与动力学适应组织成可复用低层策略；SuperMimic 额外研究可微动力学世界模型带来的训练提速。
2. **多模态与技能学习：** M3imic / MultiModalWBC、UFO、OMG。重点分别落在统一命令表示、无监督技能表征、多模态运动生成与执行跟踪器解耦。
3. **遥操作与移动操作：** TWIST2、ResMimic、OpenHLM、OpenLambda / λ₀。重点分别落在人体示范采集、载荷下的残差修正、全身原生 VLA、第一视角人类数据到机器人动作对齐。

## 原文的工程判断

- 全身控制已从表演单个固定动作扩展为可复用的“运动执行基座”；**能跟踪动作不等于能完成未知场景的操作任务**。
- 各项目的评测口径和机器人/仿真平台不同，不能直接以视频演示、训练时间或动作成功率排出统一名次。
- 所列 12 个 GitHub 入口来自文章 Ref；当前开源资产与运行入口需按官方仓库分别核查。
- 文章提及 HoloMotion + SuperMimic 在 LAFAN1 上单 RTX 5090 约 30 分钟、80% 成功率，以及 ResMimic 5.5 kg 搬运，属于**文章转述的特定实验口径**；本归档未独立复现，不能外推至任意动作、平台和负载。

## 官方参考仓库（依原文顺序）

1. <https://github.com/HorizonRobotics/HoloMotion>
2. <https://github.com/NVlabs/GR00T-WholeBodyControl>
3. <https://github.com/GalaxyGeneralRobotics/OpenTrack>
4. <https://github.com/HybridRobotics/whole_body_tracking>
5. <https://github.com/HybridRobotics/supermimic>
6. <https://github.com/Renforce-Dynamics/MultiModalWBC>
7. <https://github.com/Roboparty/UFO>
8. <https://github.com/Tsinghua-MARS-Lab/OMG>
9. <https://github.com/amazon-far/TWIST2>
10. <https://github.com/amazon-far/ResMimic>
11. <https://github.com/OpenHLM-project/OpenHLM>
12. <https://github.com/LambdaHumanoid/openlambda>

## 对 wiki 的映射

- [12 项横向比较](../../wiki/comparisons/humanoid-wbc-open-source-12-2026.md)
- [WBC 概念](../../wiki/concepts/whole-body-control.md)
- [BFM 纵深路线](../../roadmap/depth-bfm.md)

**证据分层：** 本文是二手策展资料；详细流程、已开源范围和可运行入口以 [12 项官方 README 核对归档](../repos/humanoid-wbc-12-official-readme-audit-20261123.md) 与既有独立实体/论文页为准。
