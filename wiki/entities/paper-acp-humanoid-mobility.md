---
type: entity
project_id: acp-humanoid-mobility-attention-based-contacts
tags: [paper, humanoid, locomotion, contact, reinforcement-learning]
status: complete
updated: 2026-10-23
arxiv: "2610.31608"
project: https://leggedrobotics.github.io/ACP/
related:
  - ../tasks/humanoid-locomotion.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
  - ../concepts/contact-dynamics.md
  - ../concepts/terrain-adaptation.md
sources:
  - ../../sources/papers/acp_arxiv_2610_31608.md
  - ../../sources/sites/acp-attention-based-contacts.md
  - ../../sources/repos/acp-code-availability.md
  - ../../sources/repos/ame2-minimal-leggedrobotics.md
summary: "ACP 用 13.2 分钟人体动作热启动接触规划与全身跟踪：部署时从 LiDAR、高程图和本体感知规划接触，不需要参考动作。"
---

# ACP：注意力式接触规划人形移动（arXiv:2610.31608）

**ACP**（项目页名称；论文题名为 *Humanoid Mobility via Attention-based Contacts*）通过预测机器人身体与环境的接触，将动作先验、环境观察和全身控制连接起来。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ACP | Attention-based Contacts | 项目页采用的简称；可读取论文题名未给出更长的正式展开 |
| RL | Reinforcement Learning | 微调阶段探索动作数据未覆盖的环境变化 |
| RSL | Robotic Systems Lab | leggedrobotics 项目域名所对应的研究团队简称；此处仅作来源定位 |

## 为什么重要

单纯通过奖励和探索来获得复杂的全身协调很困难；直接跟踪参考轨迹又会限制机器人适应训练数据之外的地形。ACP 把“下一步身体要在哪里接触环境”作为显式中间表示，尝试保留动作先验提供的协调性，同时让强化学习继续适应新环境。

## 核心原理

ACP 把接触作为感知与控制之间的接口，串联两个网络：

1. **通用接触规划器：** 输入 LiDAR 高程图扫描、本体感知和目标，在 1.5 秒时域内预测九个身体部位（手、肘、膝、脚和骨盆）的接触位置。
2. **通用全身跟踪器：** 接收九个三维接触点及高程图、本体感知，生成 23 个关节的动作。

项目页报告，两个网络部署时均以 50 Hz 在机载运行。13.2 分钟的人体动作只在训练期提供接触先验；强化学习进一步探索演示数据未覆盖的环境变化。部署不输入参考动作，同一策略覆盖项目页展示的多个技能。

### 流程总览

```mermaid
flowchart LR
  prior["动作数据：训练先验"] --> planner["注意力式接触规划器"]
  observation["机载观察"] --> planner
  planner --> contacts["未来身体—环境接触"]
  contacts --> tracker["接触条件全身跟踪器"]
  tracker --> action["协调的全身动作"]
  action --> environment["复杂地形"]
  environment --> observation
  planner -. "RL 微调探索新环境" .-> tracker
```

## 感知与神经地图

ACP 项目页说明其复用了 AME-2 的神经地图，配合头载 Livox MID-360。地图在 Jetson Orin NX CPU 上运行，以 10 Hz 处理约 20,000 点扫描，3 cm 网格分辨率下单次更新小于 30 ms，并与里程计和策略并行。这里的地图为接触规划提供高程图观察。

## 工程实践

项目页展示爬梯、攀台、翻越、跳下、起身、楼梯、踩石和窄梁平衡，并包含组合地形与失败恢复片段。项目页本身没有 ACP 接触规划器或跟踪器的源码/数据下载入口；其引用的 AME-2 官方代码仓库仅是前作的最小实现，README 明确神经地图模型尚未包含。因此不能把该仓库当作 ACP 完整实现或可复现的神经地图代码。详见[开放状态核查](../../sources/repos/acp-code-availability.md)及[AME-2 官方仓库归档](../../sources/repos/ame2-minimal-leggedrobotics.md)。

### 源码运行时序图

**不适用（暂无法核实作者公开了可运行代码）**。项目页尚不可读取，因此不能确认官方训练、推理或部署入口；取得官方仓库后再补充运行时序。

## 实验与评测

论文摘要称该框架面向复杂地形人形移动，并采用“接触规划器 + 条件跟踪器 + RL 微调”的组合。当前可读取的摘要未提供数值对比、机器人型号或消融结果；定量评测需以论文全文及项目页为准。

## 与其他工作对比

| 路线 | 先验与适应方式 | 主要边界 |
|------|----------------|----------|
| 参考轨迹跟踪 | 直接跟踪预先定义的动作轨迹 | 运动自然，但适应范围受训练轨迹覆盖限制 |
| 无参考 RL | 从奖励与探索中学习协调行为 | 复杂全身协调的探索和奖励设计较难 |
| ACP | 训练期动作先验 + 接触计划 + RL 微调 | 将接触设为中间接口；公开代码状态尚待核实 |

## 结论

**ACP 的核心设计是用未来身体—环境接触作为中间接口，在训练期借助动作先验、再用强化学习提升对新环境的适应能力。**

1. 复现时分别验证接触规划器和接触条件跟踪器，避免只看最终动作成功率。
2. 把训练和部署输入分开记录：动作数据提供训练先验，部署不需要参考输入。
3. 目前可读取的摘要不足以支持定量优越性结论；比较数值前先核对论文全文。
4. 项目页与代码开放状态待复核；在确认作者发布入口前，不应假定实现可直接运行。

## 局限与风险

- **能力证据范围：** 项目页展示人形移动、攀爬、翻越、起身、组合地形与恢复；当前来源未展示目标物操作任务，因此不能据此认定 whole-body manipulation 已验证。
- **接触计划质量是关键依赖：** 规划器预测偏差会传给条件跟踪器；可读取摘要未展示其误差边界或失效案例。
- **证据粒度有限：** 当前摘要没有可引用的指标、硬件型号或消融数字，本文不外推具体性能。
- **源码状态：** 项目页未列 ACP 接触规划器或全身跟踪器的下载入口；AME-2 官方仓库不含神经地图模型。后续应关注项目页是否补充发布链接。

## 关联页面

- [Humanoid Locomotion](../tasks/humanoid-locomotion.md)
- [楼梯与障碍感知移动](../tasks/stair-obstacle-perceptive-locomotion.md)
- [Contact Dynamics](../concepts/contact-dynamics.md)
- [Terrain Adaptation](../concepts/terrain-adaptation.md)
- [AME-2 神经地图与感知式移动](paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md)

## 参考来源

- [论文摘录：Humanoid Mobility via Attention-based Contacts](../../sources/papers/acp_arxiv_2610_31608.md)
- [ACP 项目页归档](../../sources/sites/acp-attention-based-contacts.md)
- [代码与数据开放状态核查](../../sources/repos/acp-code-availability.md)
- [AME-2 官方最小代码仓库](../../sources/repos/ame2-minimal-leggedrobotics.md)

## 推荐继续阅读

- [arXiv:2610.31608](https://arxiv.org/abs/2610.31608)
- [ACP 项目页](https://leggedrobotics.github.io/ACP/)
