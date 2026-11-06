---
type: entity
project_id: hlo-amp
tags: [paper, humanoid, loco-manipulation, motion-tracking, reinforcement-learning, unitree-g1]
status: draft
updated: 2026-11-05
arxiv: "2611.05113"
related:
  - ./paper-reflexarc.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/hlo_amp_arxiv_2611_05113.md
original:
  - ../../sources/papers/hlo_amp_arxiv_2611_05113.md
summary: "HLO-AMP：在物体参与的场景中训练低层运动先验，再用高层策略完成 G1 双手箱体搬运。"
---

# HLO-AMP: Hierarchical Loco-Manipulation from Object-Aware Motion Priors（基于物体感知运动先验的分层移动操作）

## 一句话定义

HLO-AMP 使用包含被操作物体的运动模仿训练低层控制器，再由高层策略组织完整的箱体搬运任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HLO-AMP | Hierarchical Loco-Manipulation from Object-Aware Motion Priors | 本论文方法 |
| MoCap | Motion Capture | 人体动作采集与重定向参考 |
| WBC | Whole-Body Control | 全身控制 |
| PPO | Proximal Policy Optimization | 策略训练算法 |

## 核心信息

| 项 | 内容 |
|----|------|
| 机构 | Robotics and AI Institute |
| 论文 | [arXiv:2611.05113v1](https://arxiv.org/html/2611.05113v1)，CoRL 2026 |
| 平台 | Unitree G1；双手搬运箱体 |
| 代码 | 论文未列官方代码入口；开放状态待进一步核实 |

## 流程总览

```mermaid
flowchart TD
  mocap["人体动作采集与重定向"] --> low["含箱体仿真的低层运动模仿训练"]
  low --> frozen["冻结低层全身控制策略"]
  goal["物体位姿与目标"] --> high["高层任务策略"]
  high --> bounds["训练数据范围内的任务空间命令"]
  bounds --> frozen
  frozen --> robot["G1 双手搬运与放置"]
```

## 核心机制

- 低层用行走、持箱行走、拾取及放置等参考动作训练，箱体真实参与仿真接触；输入是基座高度、姿态、平面速度与双末端位置等 12 维命令，输出关节目标。
- 高层依据本体、箱体面中心和目标产生低层命令；将命令限制在低层训练数据范围内。训练采用阶段奖励和重置，部署策略不依赖阶段标签。
- 低层命令训练加入噪声，并逐步移除对参考关节位置残差基准的依赖，以适配高层调用。

## 源码运行时序图

不适用：目前未核实到可运行的官方实现，不能据此绘制源码时序。

## 实验与评测

- 20 秒仿真评测：整体成功率 80.7%，重箱 70.5%；无物体 WBC 基线分别为 50.3% 和 19.7%。这些数字只适用于论文的任务与评测设置。
- 消融中去掉命令范围限制后，20 秒成功率降至 15.5%；去掉观测噪声后降至 3.0%。
- 真机展示不同箱体与掉落后恢复；7–8 kg 试验接近执行器极限，持续运行可能热停机。

## 与其他工作对比

- **实验内基线（表 2，20 秒）**：三种对照共用本文的域随机化、PPO 配置、算力与任务。单体端到端策略成功率 3.6%（91.6% 能走到抬起阶段，但学不会双手抓取）；无物体 WBC 为 50.3%（轻箱 65.7%、重箱 19.7%）；加末端力扰动替代的 WBC 为 77.1%，但掉箱率 78.0%。本文 80.7%，掉箱率 26.8%，N&S 指数 1.00（基线 1.29 / 1.16，越高越差）（§4.2）。
- **用随机力扰动代替物体的全身控制**：[FALCON](./paper-loco-manip-161-109-falcon.md) 用末端随机力课程训练上下肢双智能体，[HOMIE](./paper-loco-manip-161-040-homie.md)、[OmniH2O](./paper-hrl-stack-08-omnih2o.md) 等训练时不建模被操作物体（§2）。本文让箱体真实进入低层训练；消融中改用力替代后成功率只有 6.7%（表 3）。
- **冻结 WBC + 高层策略**：[VIRAL](./paper-viral-humanoid-visual-sim2real.md)、[DoorMan](./paper-doorman-opening-sim2real-door.md) 在预训练 WBC 上蒸馏高层策略，[DemoHLM](./paper-loco-manip-161-136-demohlm.md) 用示范条件的行为克隆；论文称这类方案会继承低层不感知物体的局限（§2）。
- **保留运动先验的桥接方法**：[VisualMimic](./paper-notebook-visualmimic.md) 同样限制下发给跟踪器的命令，最接近本文，但用两阶段蒸馏的关键点跟踪器；[PhysHSI](./paper-amp-survey-15-physhsi.md) 部署时也不需参考动作，但论文称其结果限于轻箱、未含大扰动恢复；[Pro-HOI](./paper-loco-manip-161-074-pro-hoi.md) 依赖运动学规划器提供根轨迹（§2「Positioning of our work」）。
- **奖励塑形路线**：[WoCoCo](./paper-loco-manip-161-116-wococo.md) 把长程任务拆成顺序接触阶段，需要细致的奖励工程（§2）。以上相关工作的数字均未在同一设置下复现，不能与本文表 2 横比。

## 结论

物体参与低层训练与高层命令约束共同服务于负载下的长时程双手操作。

1. 评估接触丰富控制时，应考虑真实物体对多点接触力的影响。
2. 层级组合要检验上层命令是否超出下层训练分布。
3. 真机物体位姿来自外部动捕，尚未验证完整机载感知闭环。

## 局限与待核查

- 向其他任务迁移需要新示范、阶段图和奖励；自然度指标不是用户研究。
- 原文部分附录尚待逐项核对，代码开放状态也待确认；本节点暂为草稿。

## 关联页面

- [ReflexArc：全身触觉反射](./paper-reflexarc.md)
- [机器人运动](../tasks/locomotion.md)

## 参考来源

- [论文来源归档](../../sources/papers/hlo_amp_arxiv_2611_05113.md)
- [HLO-AMP 原文 v1](https://arxiv.org/html/2611.05113v1)

## 推荐继续阅读

- [作者视频](https://youtu.be/viR_ULnRDQQ)
