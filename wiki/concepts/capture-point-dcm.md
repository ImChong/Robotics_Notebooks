# Capture Point / DCM

**Capture Point（捕获点）** 和 **DCM（Divergent Component of Motion，发散运动分量）** 是腿式机器人动态平衡与步态控制里两个非常关键的概念，用来描述系统“往哪里倒”以及“该踩到哪里才能救回来”。

## 一句话定义

- **Capture Point**：如果机器人已经在往前倒，那么要想在有限步内停住，脚应该踩到哪里。
- **DCM**：把系统运动拆成“会发散的那一部分”，方便分析动态平衡和控制。

一句话说白了：

> ZMP 更像“现在还能不能稳住”，Capture Point / DCM 更像“已经开始失稳时，下一步该怎么救”。

## 为什么重要

LIP / ZMP 是理解双足步行的经典入口，但它们有一个局限：

- 更偏静态或准动态平衡理解
- 对快速运动、强扰动恢复、高动态步态不够直观

一旦机器人已经在往前倒，光说“ZMP 要在支撑多边形里”就不够了。

这时候更核心的问题变成：

- 系统当前的发散趋势往哪里走
- 是否还能通过下一步踩踏把自己接住
- 如果能接住，脚应该放到哪里

这正是 Capture Point / DCM 的核心价值。

## 背后的模型直觉

这些概念通常建立在 **LIP（线性倒立摆）** 模型上。

LIP 的经典动力学：

$$
\ddot{x} = \omega^2 (x - x_{zmp}), \quad \omega = \sqrt{g / z_c}
$$

这里：
- \( x \)：质心位置
- \( \dot{x} \)：质心速度
- \( x_{zmp} \)：ZMP
- \( z_c \)：质心高度
- \( \omega \)：倒立摆自然频率

这个系统很重要的一点是：
- 它包含**稳定模式**和**发散模式**
- Capture Point / DCM 正是在抓“发散那一支”

## Capture Point 是什么

最经典的一维定义是：

$$
\xi = x + \frac{\dot{x}}{\omega}
$$

其中：
- \( x \)：质心位置
- \( \dot{x} \)：质心速度
- \( \omega = \sqrt{g/z_c} \)

这个量 \( \xi \) 就是 Capture Point。

它的物理意义可以理解为：

> 如果你此刻想把机器人停住，并且允许它再迈一步，那么脚应该踩向的位置大致就是这个点。

### 直觉理解

- 如果机器人站得很稳，速度很小，Capture Point 靠近 CoM
- 如果机器人往前冲得很快，Capture Point 会跑到前面很远
- 如果 Capture Point 已经跑出支撑区域，而你又不能迈步，那大概率稳不住

## DCM 是什么

DCM（Divergent Component of Motion）在很多文献里和 Capture Point 非常接近，甚至在 LIP 条件下它们本质上就是一个东西。

它定义为：

$$
\xi = x + \frac{\dot{x}}{\omega}
$$

这个变量有个关键动力学：

$$
\dot{\xi} = \omega (\xi - x_{zmp})
$$

这个式子非常有启发性。

它说明：
- DCM 自己会发散
- ZMP 可以用来“拉住”这个发散趋势
- 步态控制很多时候其实就是在控制 DCM

## 为什么 DCM 比单看 CoM 更有用

如果你只看 CoM：
- 会知道机器人现在在哪里
- 但不一定知道它“接下来会往哪里跑”

如果你看 DCM：
- 你会同时考虑位置和速度
- 更接近系统真正的失稳趋势

所以 DCM 的好处是：

> 它把“状态 + 趋势”压进了一个更适合动态平衡控制的变量里。

## Capture Point / DCM 和 ZMP 的关系

这三个概念很容易混。

### ZMP
更像：
- 当前支撑稳定性相关量
- 地面合力矩平衡点
- 常用来做经典步态规划和稳定性约束

### Capture Point / DCM
更像：
- 当前运动发散趋势
- 如果失稳，下一步该踩哪里
- 更适合做动态平衡和扰动恢复分析

一句话：

> ZMP 更像“当前还能不能稳住”，Capture Point / DCM 更像“已经开始不稳时该怎么接住”。

见：[LIP / ZMP](./lip-zmp.md)

## 和 Contact Dynamics 的关系

Capture Point / DCM 看起来像质心变量，但本质上依赖接触能力。

因为你能不能把自己接住，取决于：
- 能不能在合适时刻建立新接触
- 能不能在可行位置落脚
- 接触力和摩擦约束是否允许

所以它不是脱离 contact 单独存在的量。

见：[Contact Dynamics](./contact-dynamics.md)

## 和 Floating Base Dynamics 的关系

在 floating base 系统里，base 不可直接控制，整体稳定性非常依赖：
- 接触组织
- 质心速度
- 动量演化

Capture Point / DCM 是对这类发散趋势的简化描述。

见：[Floating Base Dynamics](./floating-base-dynamics.md)

## 在机器人中的典型应用

### 1. 扰动恢复
机器人被推一把之后：
- 是否需要迈步
- 往哪迈
- 多远能接住

这些问题几乎都可以借助 Capture Point 理解。

### 2. 动态步态控制
比起只用 ZMP，很多更动态的 walking controller 会显式或隐式使用 DCM / CP。

### 3. Footstep Planning
Capture Point 可以直接影响：
- 下一步落脚位置
- 步长调整
- 何时从站立转入迈步

### 4. 高动态 locomotion 直觉建立
如果你想理解“为什么机器人不能一直原地稳住，而必须迈步”，Capture Point 是最好的概念之一。

## 在人形控制里的意义

对于人形机器人来说：
- 站立时可以主要看静态 / 准动态平衡
- 一旦开始快速行走、跑、被推、切换支撑，动态因素立刻主导

这时：
- Capture Point 告诉你“还能不能接回来”
- DCM 告诉你“发散分量现在往哪里走”

所以在现代 humanoid locomotion 语境里，它们往往比单独讲 ZMP 更贴近控制直觉。

## 常见误区

### 1. 以为 Capture Point 能完全替代 ZMP
不能。它们关注的物理问题不同，常常是互补关系。

### 2. 以为 Capture Point 是精确万能落脚解
不是。它是很有力的简化指标，但真实系统还要考虑接触、关节限位、可达性、时序等。

### 3. 以为 DCM 是一个完全不同于 Capture Point 的新东西
在很多 LIP 语境里，它们本质上非常接近。区别更多在使用语境和控制表达方式。

### 4. 低估速度项的重要性
很多初学者只盯着 CoM 位置，但动态平衡真正关键的是“位置 + 速度”的组合。

## 推荐使用建议

### 如果你学 locomotion
一定值得掌握。因为它能把“为什么要迈步”这件事讲得非常直观。

### 如果你学 MPC / footstep planning
Capture Point / DCM 是很重要的中间概念。

### 如果你学 WBC / TSID
哪怕你不直接用它们做控制变量，也建议理解，因为它们能帮你建立更动态的稳定性直觉。

## 推荐继续阅读

- Pratt et al., *Capture Point: A Step toward Humanoid Push Recovery*
- Englsberger et al., *Three-Dimensional Bipedal Walking Control Based on Divergent Component of Motion*
- [LIP / ZMP](./lip-zmp.md)
- [Contact Dynamics](./contact-dynamics.md)

## 一句话记忆

> Capture Point / DCM 研究的是“机器人当前的发散趋势往哪里走，以及下一步该踩到哪里才能把自己接住”，它是比 ZMP 更动态的平衡与扰动恢复理解框架。
