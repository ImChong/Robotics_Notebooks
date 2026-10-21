---
type: overview
tags: [overview, vla, world-action-models, manipulation, dexterous, tactile, navigation, digest]
status: complete
updated: 2026-10-17
related:
  - ../methods/vla.md
  - ../concepts/world-action-models.md
  - ./embodied-papers-2026-10-15-31-digest-map.md
  - ../entities/paper-pacer.md
sources:
  - ../../sources/blogs/wechat_embodied_station_robot_36_papers_2026-10-17.md
  - ../../sources/raw/wechat_embodied_station_robot_36_papers_2026-10-17.md
summary: "2026-10-17 具身智能小站汇总的 36 篇新论文索引：按主题把每篇映射到独立详情节点（30 篇新建、6 篇沿用既有节点），标注文章阅读优先级；开源状态以各详情页为准。"
---

# 2026-10-17 具身新论文 36 篇：技术地图索引

> 本页是**索引**，不替代各论文详情页。每篇论文一个独立节点，方法、实验与开源核查见对应页面。

## 一句话定义

把公众号单日汇总的 36 篇具身智能论文，按「纠错与长程执行 / 世界动作模型与评测 / 接触与灵巧操作 / 人类视频与人形迁移 / 导航与感知 / 规划、智能体与多机」归类，每篇一个独立详情节点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| WAM | World Action Model | 联合预测视频与动作的世界-动作模型 |
| SFT | Supervised Fine-Tuning | 监督微调 |
| BC | Behavior Cloning | 行为克隆 |
| RL | Reinforcement Learning | 强化学习 |

## 主题分组（文章阅读优先级：深读 1 / 跟进 2 / 扫读 33）

### 纠错、长程执行与策略改进

- [PACER](../entities/paper-pacer.md)（2610.19248，**深读**）— 人工纠错的过程感知重加权
- [LoopWAM](../entities/paper-loopwam.md)（2610.18873）— 部署轨迹闭环改进
- [ReEnvision](../entities/paper-reenvision.md)（2610.17977）— 只修复受偏差影响的动作
- [Orchestra](../entities/paper-orchestra.md)（2610.18352）— 规划与动作生成分工的长序列操作
- [LADDER](../entities/paper-ladder-long-horizon-benchmark.md)（2610.18053）— 长程操作基准
- [TD-Aug](../entities/paper-td-aug.md)（2610.19129）— 想象目标增强行为克隆

### 世界动作模型、世界模型与评测

- [R²-WAM](../entities/paper-r2-wam.md)（2610.18038）— 无历史数据的生成回放持续学习
- [ForceWAM](../entities/paper-forcewam.md)（2610.18852）— 示范与自由探索数据解耦
- [WoVR+](../entities/paper-wovr-plus.md)（2610.17922）— 世界模型作可靠仿真器
- [D3WM](../entities/paper-d3wm.md)（2610.19212）— 漂移-扩散解耦的未来帧预测
- [RoboFQE](../entities/paper-robofqe.md)（2610.18699）— 离线轨迹的免 rollout 策略评估
- [SonoWorld](../entities/paper-sonoworld.md)（2610.19327）— 位姿可控超声世界模型

### 接触、力觉与灵巧操作

- [CTP](../entities/paper-ctp.md)（2610.18848，**跟进**）— 单示范接触轨迹提示
- [FACT](../entities/paper-fact-egoforce-human-dexterity.md)（2610.19155）— 人类力-动作-轨迹基础模型
- [ILA](../entities/paper-ila-interaction-latents-as-actions.md)（2610.18855）— 交互潜变量作为动作
- [ForeTac](../entities/paper-foretac.md)（2610.18118）— 动作接触后果预测引导
- [DexMimic](../entities/paper-dexmimic.md)（2610.19089）— 价值引导重置的灵巧手跟踪
- [PACT](../entities/paper-pact-hand-codesign.md)（2610.18443）— 手结构与重定向协同设计

### 人类视频、人形迁移与仿真

- [λ₀](../entities/paper-lambda0-egocentric-human-pretraining.md)（2610.18950）— 第一视角人类数据到人形全身移动操作
- [EgoMatrix](../entities/paper-egomatrix.md)（2610.18822）— 第一视角视频生成双手示范
- [Steertuning](../entities/paper-steertuning.md)（2610.18711）— 少量人类视频引导策略
- [TRACE](../entities/paper-trace-humanoid-transfer.md)（2610.17999）— 跨人形的全身控制迁移
- [RoboSimStudio](../entities/robosimstudio.md)（2610.18835）— 交互式 Real2Sim 工具

### 导航、感知与数据

- [CoSMO](../entities/paper-cosmo.md)（2610.18321，**跟进**）— 保形语义地图的安全导航
- [NavAble](../entities/paper-navable.md)（2610.19330）— 无障碍导航数据集
- [GLST](../entities/paper-glst.md)（2610.17927）— 链式目标在线导航记忆
- [MIND](../entities/paper-mind-incremental-grounding.md)（2610.19262）— 指令条件增量接地
- [I2O2I-Nav](../entities/paper-i2o2i-nav.md)（2610.18307）— 室内-室外-室内导航基准
- [RTISSLAM](../entities/paper-rtisslam.md)（2610.18479）— 成像声呐 SLAM

### 智能体、规划与多机

- [EmbodiedEvo](../entities/paper-embodied-evo-coding-agent.md)（2610.18867）— 自演化代码技能
- [SEAR](../entities/paper-sear.md)（2610.18792）— 自演化智能体机器人的对照评测
- [Physical RSI](../entities/physical-rsi-research-map.md)（2610.18842）— 物理递归自我改进研究地图
- [APTree](../entities/paper-aptree.md)（2610.18523）— 建筑机器人行为树规划
- [MEND](../entities/paper-mend.md)（2610.18850）— 空地协同神经符号扩散规划
- [CALM](../entities/paper-calm-decentralized-aerial-manipulation.md)（2610.18425）— 多机协同空中搬运
- [Cepheid](../entities/paper-cepheid.md)（2610.18871）— 跨形态运动补全

## 读法提醒

- 「深读 / 跟进 / 扫读」分档是公众号编辑的判断，不是本库评级。
- 多数论文在入库日项目页标注「即将发布」或未列代码，复现前以各详情页的开源核查为准。
- Physical RSI 沿用既有研究地图节点（其仓库即论文配套资源），未另建重复页。

## 关联页面

- [VLA](../methods/vla.md) — VLA 方法概览
- [World Action Models](../concepts/world-action-models.md) — WAM 相关论文的概念入口
- [2026-10-15 具身新论文 31 篇索引](./embodied-papers-2026-10-15-31-digest-map.md) — 上一期

## 参考来源

- [公众号汇总归档](../../sources/blogs/wechat_embodied_station_robot_36_papers_2026-10-17.md)
- [原始抓取](../../sources/raw/wechat_embodied_station_robot_36_papers_2026-10-17.md)

## 推荐继续阅读

- [原文（具身智能小站）](https://mp.weixin.qq.com/s/I4XP5JSG9QNmm4hd1nk2JQ)
