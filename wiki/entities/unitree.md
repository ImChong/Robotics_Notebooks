---
type: entity
---

# Unitree

**Unitree Robotics（宇树科技）** 是当前腿式机器人和人形机器人领域最有影响力的公司之一。

## 一句话定义

如果说很多论文和算法都在讲“机器人应该怎么走、怎么跑、怎么控制”，那 **Unitree** 的重要性在于：

> 它把高性能腿式平台和人形平台真正做成了更多团队能买到、能部署、能调算法的实际硬件入口。

一句话说白了：

> `Unitree` 在当前机器人圈的意义，不只是一个公司，而是把腿式机器人和人形机器人的真实研究门槛大幅拉低的一条关键硬件主线。

## 为什么重要

以前很多机器人研究会卡在一个很现实的问题上：

- 仿真里能做
- 论文里能讲
- 但真机平台买不到、太贵、维护成本太高

Unitree 重要的地方就在于：

- 它把四足和人形平台做成了更可获得的硬件入口
- 它让更多高校、实验室、创业团队能真正接触到动态腿式机器人
- 它在 RL、locomotion、sim2real 这些方向里，实际成为了大量实验落地的硬件载体
- 它的影响力已经不只是产品，而是“研究平台普及化”

## 它到底是什么

### 1. 不是算法框架
Unitree 不是 MuJoCo、Isaac Gym、Pinocchio、Crocoddyl 这种软件工具链。

它不是：
- 仿真器
- 优化库
- 控制框架

它更像：
- 机器人硬件平台提供者
- 四足 / 人形产品生态节点
- 真实部署和 sim2real 的主要落点之一

### 2. 它是“算法真正落地的那层硬件”
很多技术栈页面都在讲：
- trajectory optimization
- MPC
- WBC
- RL
- sim2real

但这些东西最终都要落在具体机器人平台上。

Unitree 的价值就在于：

> 它是这些方法最常见的真实硬件承载体之一。

## Unitree 当前最值得关注的产品语境

### 1. 四足线
Unitree 最早广泛出圈的是四足机器人。

这一条线的重要性在于：
- 为 locomotion、强化学习、扰动恢复、sim2real 提供了稳定研究平台
- 让很多本来只能在论文里看的东西，变成可真实验证的实验系统

### 2. 人形线
Unitree 现在已经明显不只是四足公司。

当前在人形机器人语境里，最常被讨论的是：
- **G1**
- **H1 / H1-2**

这一点非常关键，因为它说明：

> Unitree 已经从“腿式机器人硬件公司”进一步变成“人形机器人研究与应用平台的重要提供者”。

## Unitree 为什么对当前项目主线很重要

当前 `Robotics_Notebooks` 的主线是：

```text
LIP / ZMP
  ↓
Centroidal Dynamics
  ↓
Trajectory Optimization / MPC
  ↓
TSID / WBC
  ↓
State Estimation / System Identification / Sim2Real
```

这条主线最终必须落到真实平台。

Unitree 在这里的重要性是：
- 它给了真实的四足 / 人形平台
- 它让 sim2real 不再只是概念
- 它让“控制 / 学习 / 优化”这些方法真正有了硬件落点

一句话：

> 在这条技术链里，Unitree 不负责“怎么想”，它负责“最终在哪台真实机器上验证”。

## Unitree 和 RL / locomotion 的关系

### 1. RL 的真实平台
如果你做人形 / 四足 RL，最终很常见的目标就是：
- 先在 MuJoCo / Isaac Gym / Isaac Lab 里训练
- 再迁移到 Unitree 这类平台

所以 Unitree 在很多研究链路里就是 sim2real 的终点之一。

### 2. Locomotion 的真实平台
很多 locomotion 研究，如果只停在仿真，很难说明方法真正靠谱。

而 Unitree 平台之所以重要，是因为：
- 动态腿式动作可以真实验证
- 扰动恢复、地形适应可以做真实实验
- 能让你看见“仿真策略落到硬件后还剩下多少问题”

## Unitree 和 WBC / MPC / TSID 的关系

如果你做传统控制路线：
- Unitree 平台经常是 WBC / MPC / locomotion controller 的真实验证平台

如果你做 learning-based 路线：
- 它又是 sim2real / RL policy 部署的硬件落点

所以它不是绑定某一种方法，而是方法最终汇合的地方。

见：[Whole-Body Control](../concepts/whole-body-control.md)

见：[Model Predictive Control (MPC)](../methods/model-predictive-control.md)

见：[Reinforcement Learning](../methods/reinforcement-learning.md)

## 为什么它在当前行业里影响这么大

### 1. 低门槛硬件普及
过去很多高动态平台只有顶级实验室能碰。

Unitree 的意义之一就是：
- 更可买
- 更可部署
- 更容易形成社区与复现生态

### 2. 把“研究平台”变成“更大众可获得平台”
这对整个机器人研究生态影响很大。

因为它改变的不是某篇论文，而是：
- 谁能做实验
- 谁能做真实机器人验证
- 谁能把算法从仿真带到硬件

### 3. 人形线把影响力继续往上推
从四足扩到人形之后，Unitree 的影响已经不只是“狗机器人平台”，而是当前人形研究生态的重要参与者。

## 常见误区

### 1. 以为 Unitree 只是做四足的
这已经过时了。现在它在人形语境里的影响也很强。

### 2. 以为有了 Unitree 平台，sim2real 就简单了
错。平台解决的是“有真机可上”，不等于状态估计、系统辨识、控制延迟、观测噪声这些问题自然消失。

### 3. 以为 Unitree 是算法主线
不是。它是硬件平台主线。真正的算法主线还是控制、优化、学习和 sim2real。

### 4. 以为硬件平台和软件工具链是替代关系
不是。更准确地说：
- MuJoCo / Isaac Lab / Pinocchio / Crocoddyl 是工具链
- Unitree 是落地平台

## 推荐使用建议

### 如果你做仿真研究
也值得关注 Unitree，因为你设计任务和 observation / action space 时，最好提前想清楚未来是不是可能迁到这类真实平台。

### 如果你做 sim2real
Unitree 是非常重要的目标平台语境，值得尽早理解它的硬件约束和部署现实。

### 如果你做人形控制
Unitree 之所以重要，不只是因为它火，而是因为它让人形控制研究真正更容易接上真实硬件。

## 推荐继续阅读

- 官方网站：<https://www.unitree.com/>
- 官方 GitHub：<https://github.com/unitreerobotics>
- G1 页面：<https://www.unitree.com/g1/>
- H1 页面：<https://www.unitree.com/h1/>

## 参考来源

- 官方网站：<https://www.unitree.com/>
- 官方 GitHub：<https://github.com/unitreerobotics>
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (2021) — 基于 Unitree 的 sim2real 代表工作

## 关联页面

- [legged_gym](./legged-gym.md)
- [Locomotion](../tasks/locomotion.md)
- [Humanoid Hardware](../../references/papers/humanoid-hardware.md)

## 一句话记忆

> Unitree 的核心意义，不只是做出四足和人形机器人，而是把真实腿式 / 人形平台变成更多团队能真正拿来做控制、RL、locomotion 和 sim2real 研究的硬件入口。
