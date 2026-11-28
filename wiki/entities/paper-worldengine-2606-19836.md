---
type: entity
project_id: driving-worldengine
tags: [paper, autonomous-driving, vla]
status: complete
updated: 2026-11-28
arxiv: "2606.19836"
code: https://github.com/OpenDriveLab/WorldEngine
summary: "WorldEngine：高保真闭环仿真与长尾驾驶强化后训练。真实日志经MTGS三维高斯重建形成SimEngine交互环境，AlgEngine中的VADv2闭环策略产生反馈，失败样本筛选后用于安全约束强化后训练；论文提出Behavior World Model。"
original:
  - ../../sources/papers/ad_top10_2026_worldengine.md
sources:
  - ../../sources/papers/ad_top10_2026_worldengine.md
  - ../../sources/repos/ad_top10_2026_worldengine.md
  - ../../sources/blogs/wechat_shenlan_ai_autonomous_driving_top10_2026.md
related:
  - ../overview/autonomous-driving-open-source-top10-2026.md
  - ../methods/vla.md
  - ../methods/generative-world-models.md
---

# WorldEngine：高保真闭环仿真与长尾驾驶强化后训练

**英文：** World Engine: Towards the Era of Post-Training for Autonomous Driving

## 一句话定义

真实日志经MTGS三维高斯重建形成SimEngine交互环境，AlgEngine中的VADv2闭环策略产生反馈，失败样本筛选后用于安全约束强化后训练；论文提出Behavior World Model。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| 3DGS | 3D Gaussian Splatting | 道路场景三维重建 |
| E2E | End-to-End | 从观测到规划的端到端模型 |
| RL | Reinforcement Learning | 奖励信号驱动的优化 |
| VLA | Vision-Language-Action | 视觉语言动作联合学习 |

## 为什么重要

高保真闭环仿真与长尾驾驶强化后训练专门针对单纯模仿驾驶轨迹难以解释或处理长尾交互的限制，提供可核查的论文、权重或代码入口。

## 核心原理

真实日志经MTGS三维高斯重建形成SimEngine交互环境，AlgEngine中的VADv2闭环策略产生反馈，失败样本筛选后用于安全约束强化后训练；论文提出Behavior World Model。

## 流程总览

```mermaid
flowchart LR
  Logs[真实车辆日志] --> MTGS[3DGS场景重建]
  MTGS --> Sim[SimEngine]
  Model[VADv2驾驶模型] --> Sim
  Sim --> Reward[安全与驾驶反馈]
  Reward --> Hard[困难样本抽取]
  Hard --> Train[AlgEngine强化微调]
  Train -.更新.-> Model
```

此流程根据 [原始论文记录](../../sources/papers/ad_top10_2026_worldengine.md)，非跨项目通用套图。

## 源码运行时序图

```mermaid
sequenceDiagram
  autonumber
  actor Dev as 开发者
  participant CLI as scripts/closed_loop_test.sh
  participant Sim as projects/SimEngine/scripts/run_testing.sh
  participant Config as AlgEngine e2e_vadv2_50pct.py
  participant CKPT as e2e_vadv2_50pct_ep8.pth
  participant World as SimEngine闭环场景
  Dev->>CLI: 设置WORLDENGINE_ROOT后运行
  CLI->>Sim: 配置、模型权重、navtest_failures、NR
  Sim->>Config: 读取VADv2模型配置
  Sim->>CKPT: 加载预训练权重
  loop 逐时间步规划
    Sim->>World: 根据ego动作推进环境
    World-->>Sim: 渲染观测和碰撞状态
  end
  Sim-->>Dev: 288个稀有场景闭环结果与PDMS
```

按官方README和 [源码入口记录](../../sources/repos/ad_top10_2026_worldengine.md) 核对；**未在本次 ingest 执行完整 GPU 推理或训练**。

## 评测与指标

仓库Quick Test对288个稀有场景进行闭环测试，读取VADv2 50%数据训练的模型并算NAVSIM v1 PDMS。

## 工程实践

**官方发布复核（2026-11-28）：** 公开 Quick Test 用既有 VADv2 检查点在 navtest_failures 上执行闭环；官方已发布稳定检查点，并在 2026-10-27 更新 navtrain/navtest 缓存。Quick Test 脚本默认读 `ckpts/e2e_vadv2_50pct_ep8.pth`，发布目录实际在 `ckpts/vadv2/` 下；保留原目录下载时，先令脚本路径与实际文件一致。论文完整的 Behavior World Model 集成仍是仓库未完成项，不能用 Quick Test 结果冒充论文全量后训练复现。

官方仓库：https://github.com/OpenDriveLab/WorldEngine

1. ```export WORLDENGINE_ROOT=$(pwd)```
2. ```bash scripts/closed_loop_test.sh```
3. ```bash scripts/multigpu_closed_loop_test.sh```

## 结论

**高保真闭环仿真与长尾驾驶强化后训练** 是驾驶研究中一条具有独立问题设定的复现路线。

1. 先使用官方公开checkpoint与现有评测脚本建立基线。
2. 明确划分论文提出的完整系统和当前代码实际已开放的部分。
3. 记录数据、版本、硬件与指标，避免不同评测设定的数字错比。

## 与其他工作对比

- **对比预训练规模扩容与多种后训练范式（nuPlan 稀有安全关键基准，Table 1）：** 在 50k 场景 base 模型（common/rare 开环 PDMS 85.64 / 47.14，rare 闭环 SR 73.66% / PDMS* 60.98）上，仅用常规日志做后训练反而令 rare 闭环 SR 降至 69.63%；用静态稀有日志或固定稀有合成回放虽把 SR 提至 73.35% / 87.20%，但后者的自车进度 EP 跌至 32.49；去掉 Behaviour WM 的反应式闭环 rollout 取得 77.96% SR / 67.33 PDMS*；完整 **World Engine**（3DGS SimEngine + Behaviour WM 交通交互增强 + 安全约束 RL）达到 **88.89% SR / 70.12 PDMS***（同时 common 开环升至 88.95），增益超过将预训练数据从 50k 扩至 103k 的曲线趋势。
- **对比传统数据引擎、仿真回放与静态 RL 后训练（Appendix A.3）：** 既有数据引擎多停留在开环数据增广或离线压力评测，难以耦合策略自身的演化弱点；World Engine 把三维高斯重建、反应式交通行为世界模型与闭环强化后训练串成统一闭环。但相比纯粹闭环轨迹评测框架（如 [ReCogDrive](./paper-recogdrive-2506-08052.md) 使用的 NAVSIM/CARLA），World Engine 公开仓库的 Behaviour World Model 集成仍标待办，且 3DGS 渲染不自动保证物理接触与力学真实性。

## 局限与风险

官方README把Behavior World Model integration仍列为待办；simengine和algengine需分开装依赖；3DGS重建画面不保证物理交互真实。

真实上路安全和合规均不能由离线分数、仿真结果或开源状态单独证明。

## 关联页面

- [2026 自动驾驶开源项目技术地图](../overview/autonomous-driving-open-source-top10-2026.md)
- [VLA](../methods/vla.md)
- [生成式世界模型](../methods/generative-world-models.md)

## 参考来源

- [官方代码、发布资产与固定版本核对](../../sources/repos/autonomous-driving-official-repo-refresh-2026-11-28.md)

- [论文核对记录](../../sources/papers/ad_top10_2026_worldengine.md)
- [官方源码入口](../../sources/repos/ad_top10_2026_worldengine.md)
- [深蓝AI策展原文](../../sources/blogs/wechat_shenlan_ai_autonomous_driving_top10_2026.md)

## 推荐继续阅读

- [原论文](https://arxiv.org/html/2606.19836)
- [官方代码](https://github.com/OpenDriveLab/WorldEngine)
