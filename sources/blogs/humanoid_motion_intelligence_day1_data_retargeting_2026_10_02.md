# 具身智能从入门到精通 Day 1：数据与重定向

- **作者：** Yuanxq（具身智能研究室）
- **发表：** 2026-10-02
- **文章：** <https://mp.weixin.qq.com/s/9Gh-3hxglD2Zw30DCva1pQ>
- **原项目：** <https://github.com/RealXiaoze/humanoid-motion-intelligence>
- **归档依据：** 用户提供的 30 页 PDF；以下是跨主题索引和逐篇独立详情入口，不转录原文。

## 主线

视频恢复人体与物体运动 → 形态/接触/时间重定向 → 物理可执行性修正 → 场景与数据扩增 → 下游跟踪/操作。人体姿态恢复的误差、机器人关节极限和接触动力学不能用一个“重定向成功率”概括。下表每篇仅指向**一个**现有或新建的详情节点；同缩写、同方向的不同论文不合并。

| # | 论文 | 独立详情节点 | 在数据链中的作用 |
|---:|------|--------------|------------------|
| 1 | DexMV | [DexMV](../../wiki/entities/paper-dexmv.md) | 视频到灵巧手示范 |
| 2 | WHAM | [WHAM](../../wiki/entities/wham-world-human-motion.md) | 世界系人体运动恢复 |
| 3 | TRAM | [TRAM](../../wiki/entities/paper-tram-global-human-motion.md) | 相机与人体全局轨迹 |
| 4 | GVHMR | [GVHMR](../../wiki/entities/gvhmr.md) | 重力视角运动恢复 |
| 5 | PHC | [PHC](../../wiki/entities/phc.md) | 物理人体角色跟踪 |
| 6 | Retargeting Matters | [Retargeting Matters](../../wiki/entities/paper-hrl-stack-01-retargeting_matters.md) | GMR 重定向质量 |
| 7 | OmniRetarget | [OmniRetarget](../../wiki/entities/paper-hrl-stack-03-omniretarget.md) | 保留人与场景交互 |
| 8 | DynaRetarget | [DynaRetarget](../../wiki/entities/paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md) | 动力学可行轨迹 |
| 9 | Make Tracking Easy | [Make Tracking Easy](../../wiki/entities/paper-hrl-stack-02-make_tracking_easy.md) | 神经动作重定向 |
| 10 | HumanoidMimicGen | [HumanoidMimicGen](../../wiki/entities/paper-humanoidmimicgen.md) | 全身规划生成数据 |
| 11 | ECHO-G | [ECHO-G](../../wiki/entities/paper-echo-g-cospeech-humanoid.md) | 语音驱动全身动作 |
| 12 | OTRetarget | [OTRetarget](../../wiki/entities/paper-otretarget.md) | 联合机器人与物体轨迹 |
| 13 | PRISM / Counterfactual Video Generation | [PRISM](../../wiki/entities/paper-prism-real2sim2real.md) | 反事实视觉数据扩增 |
| 14 | Dense Temporal Motion Retargeting | [Dense Temporal Retargeting](../../wiki/entities/paper-dense-temporal-motion-retargeting.md) | 动作相位联合优化 |
| 15 | GestAdapt | [GestAdapt](../../wiki/entities/paper-gestadapt.md) | 工作空间约束手势 |
| 16 | HOI-Retarget | [HOI-Retarget](../../wiki/entities/paper-hoi-retarget.md) | 人物接触重定向 |
| 17 | BeyondRetarget | [BeyondRetarget](../../wiki/entities/paper-beyondretarget-monocular-humanoid.md) | 单目直接生成机器人参考 |
| 18 | MATE | [MATE](../../wiki/entities/paper-mate-virtual-teleop.md) | 多人协作遥操作数据 |
| 19 | PhyVisGen | [PhyVisGen](../../wiki/entities/paper-phyvisgen.md) | 物理与视觉一致的操作数据 |
| 20 | Automatic Labelling | [Automatic Labelling](../../wiki/entities/paper-automatic-labelling-bimanual-mobile.md) | 双臂移动操作标注 |
| 21 | HIGenNTO | [HIGenNTO](../../wiki/entities/paper-higennto-noise-space-optimization.md) | 场景交互动作生成 |
| 22 | Unified Motion Retargeting | [UMR](../../wiki/entities/paper-umr-unified-motion-retargeting.md) | 表面点云对应 |
| 23 | AnyWorld | [AnyWorld](../../wiki/entities/paper-anyworld.md) | 跨本体第一视角世界模型 |
| 24 | HiPHI | [HiPHI](../../wiki/entities/paper-hiphi.md) | 人-物交互动捕基准 |
| 25 | R2S-EGO | [R2S-EGO](../../wiki/entities/paper-r2s-ego.md) | 稀疏采集建场景 |
| 26 | Shooting for Contact | [Shooting for Contact](../../wiki/entities/paper-shooting-for-contact.md) | 接触隐式动力学修正 |
| 27 | Emergent Transfer | [Emergent Transfer](../../wiki/entities/paper-emergent-transfer-cross-config.md) | 旧本体数据迁移阈值 |
| 28 | Data Pyramid | [Data Pyramid](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md) | 数据来源综述 |
| 29 | EgoExoMoCap | [EgoExoMoCap](../../wiki/entities/paper-egoexomocap.md) | 自我/外部视角动捕 |
| 30 | EgoHTR | [EgoHTR](../../wiki/entities/paper-egohtr.md) | 地形和人体 4D 示范 |

## 维护说明

本次查重按**论文标题和 arXiv ID**，而非仅按缩写；原有 22 个节点保留，8 个缺失节点分别补齐。原文的实验数字是线索，具体比较须回到论文与项目页核查。文章本身不是任何一篇论文的官方出处。
