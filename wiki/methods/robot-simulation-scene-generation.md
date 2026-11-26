---
type: method
tags: [robotics, simulation, data]
status: complete
updated: 2026-11-26
summary: "将布局、资产、语义和任务约束编译为可重置、可执行、可复核的仿真场景。"
sources:
  - ../../sources/papers/robocasa_scene_generation_reading_arxiv_2406_02523.md
original:
  - ../../sources/papers/robocasa_scene_generation_reading_arxiv_2406_02523.md
related:
  - ../concepts/robot-simulation-three-layers.md
  - generative-data-augmentation.md
  - ../entities/paper-notebook-robocasa-large-scale-simulation-of-everyday-task.md
  - sim2real-transfer-workflow.md
---

# Robot Simulation Scene Generation · 机器人仿真场景生成

## 一句话定义

将布局、资产、语义和任务约束编译为可重置、可执行、可复核的仿真场景。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| MJCF | MuJoCo XML Configuration Format | MuJoCo 模型和场景描述 |
| LLM | Large Language Model | 可帮助提出复合任务候选 |
| BC | Behavior Cloning | 使用仿真示范做行为克隆 |

## 为什么重要

视觉效果只是其中一层。厨房看起来真实，门仍可能缺碰撞体、把手机构或可达的操作位；这样的场景不能成为可靠训练样本。

## 主要技术路线（核心机制）

分别管理空间布局、外观风格、可交互设备和对象语义。RoboCasa v1 区分布局与风格，用资产库装配厨房；论文的 120 个初版场景与 100 项任务不是后续 RoboCasa365 的发布数字。其复合任务由语言模型辅助提出，再由人筛选并实现成功条件；示范扩展是再下一层。

```mermaid
flowchart LR
    A["布局、样式和种子"] --> B["实例化家具和对象"]
    B --> C["挂接碰撞、关节、语义标签"]
    C --> D["添加任务、初始状态及成功判据"]
    D --> E["稳定性、可达性和重置检查"]
    E --> F["轨迹采集／评测快照"]
```

## 工程实践

1. 给每种资产留下尺度、质量/惯量、碰撞代理、关节限位、语义与许可记录；视觉网格和碰撞网格不必相同。
2. 对生成布局检查初始穿透、掉落、柜门开闭和机械臂可达性；对固定种子重置，验证任务判据会在正确动作后触发且不会提前触发。
3. 分开按布局、物体家族和任务切分，防止同一资产变换后的近重复流入训练和测试；日志保存场景配置以及无法生成、不可达或失败的种子。

## 局限与风险

碰撞与接触检查不能证明真机动力学一致；基于语言描述生成任务也不能代替明确定义成功条件。保真与生成速度需按目标任务选型。

## 关联页面

- [机器人仿真三层](../concepts/robot-simulation-three-layers.md)
- [生成式数据增强](generative-data-augmentation.md)
- [RoboCasa 论文](../entities/paper-notebook-robocasa-large-scale-simulation-of-everyday-task.md)
- [Sim2Real 迁移工作流](sim2real-transfer-workflow.md)

## 参考来源

- [RoboCasa v1 一手资料归档](../../sources/papers/robocasa_scene_generation_reading_arxiv_2406_02523.md)

## 推荐继续阅读

- [RoboCasa v1 原文](https://arxiv.org/html/2406.02523v1)
