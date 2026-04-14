# System Identification

**System Identification（系统辨识 / SysID）**：通过实验数据估计机器人动力学、执行器、摩擦、延迟等模型参数，使模型更接近真实系统的过程。

## 一句话定义

State Estimation 在回答“机器人现在是什么状态”，那 **System Identification** 回答的是：

> 我拿来做控制和仿真的这个模型，到底准不准？

## 为什么重要

机器人里大量方法都默认你至少有一个“还算靠谱”的模型。

比如：
- MPC 需要预测模型
- TSID / WBC 需要动力学模型
- Sim2Real 需要仿真模型
- State Estimation 也常常需要运动学 / 动力学先验

但真实机器人不是论文里的理想刚体。

真实世界里会有：
- 质量参数不准
- 转动惯量不准
- 电机带宽有限
- 摩擦模型不准
- 传动间隙 / 柔顺性
- 控制延迟
- 接触模型偏差

如果这些东西差太多，控制器就会建立在错误世界观上。

一句话：

> SysID 的作用，就是把“我以为机器人是这样的”尽量修正成“机器人其实是这样的”。

## 到底在辨识什么

System Identification 不是只辨识一个质量参数，它可能覆盖多个层级。

### 1. Rigid-body Parameters
刚体参数，包括：
- link 质量
- 质心位置
- 转动惯量

这类参数通常影响：
- inverse dynamics
- WBC / TSID
- 动力学仿真

### 2. Actuator Dynamics
执行器层参数，包括：
- 电机时间常数
- 扭矩常数
- 带宽
- 控制延迟
- 电流环 / 速度环等效动态

这类参数对 sim2real 很关键。

### 3. Friction / Damping / Compliance
非理想因素，包括：
- 库仑摩擦
- 黏性摩擦
- 关节阻尼
- 柔顺性 / 弹性
- 齿隙 / backslash / hysteresis

这些往往才是“仿真里能跑，真机一塌糊涂”的罪魁祸首。

### 4. Contact-related Parameters
接触相关参数，包括：
- 地面摩擦系数
- 接触刚度 / 阻尼
- 足底模型
- 碰撞恢复参数

在 locomotion 和 sim2real 里，这类参数很敏感。

## 基本思路

最常见的系统辨识思路是：

### 1. 设计激励实验
让机器人执行一些能“激发系统特性”的动作。

比如：
- 单关节扫频
- 正弦轨迹跟踪
- 多关节组合动作
- 站立扰动实验
- 跌落 / 接触测试（谨慎）

如果激励不够，参数就不可辨识。

### 2. 采集输入输出数据
采集：
- 控制输入
- 关节位置 / 速度 / 加速度
- IMU
- 力传感器
- 电流 / 扭矩估计

### 3. 建立参数化模型
例如：

$$
y = f(x, u; \theta)
$$

其中 \( \theta \) 是待辨识参数。

### 4. 优化参数
通过最小化预测和真实数据误差来求参数：

$$
\theta^* = \arg\min_{\theta} \sum_k \| y_k^{meas} - y_k^{pred}(\theta) \|^2
$$

这本质上就是一个参数估计问题。

## 常见方法

### 1. Least Squares / Linear Regression
如果模型能整理成对参数线性：

$$
Y(q, \dot{q}, \ddot{q}) \theta = \tau
$$

那就可以直接用最小二乘。

这是经典刚体动力学参数辨识路线。

优点：
- 清晰
- 快
- 理论成熟

缺点：
- 对噪声、未建模动态、参数耦合较敏感

### 2. Nonlinear Optimization
当模型对参数是非线性的，就要做非线性最优化。

常见于：
- actuator model fitting
- friction model fitting
- contact model fitting

### 3. Frequency-domain Identification
对执行器、伺服、低层控制链路常用。

比如扫频后估计：
- 带宽
- 相位延迟
- 共振特性

### 4. Black-box Identification
不强依赖物理结构，直接学输入输出映射。

例如：
- ARX / ARMAX
- neural network residual model
- Gaussian process residual dynamics

优点：
- 拟合能力强

缺点：
- 可解释性差
- 泛化和安全性要小心

### 5. Residual / Hybrid Modeling
工程上很实用的一种折中：
- 主体用 physics-based model
- 剩余误差用 learned residual 去补

这条路在 sim2real 和 high-performance control 里很常见。

## 在机器人中的典型用途

### 1. 提升仿真可信度
给 MuJoCo / Isaac / 自研仿真器填入更真实参数，减少 sim-real gap。

### 2. 提升模型控制表现
MPC、inverse dynamics、TSID、WBC 都会受益于更准的模型。

### 3. 帮助状态估计
更准的模型会让预测步骤更靠谱。

### 4. 支持域随机化范围设定
Domain Randomization 不是乱抖参数。

更合理的做法是：
- 先通过 SysID 估一个参数中心值
- 再围绕这个中心值做随机化

这比纯拍脑袋随机稳得多。

## 和已有页面的关系

### 和 Sim2Real 的关系
System Identification 是 sim2real 的主力方法之一，用来减少物理参数、执行器、接触模型的 gap。

见：[Sim2Real](./sim2real.md)

### 和 State Estimation 的关系
状态估计回答“当前状态是什么”，系统辨识回答“预测状态用的模型对不对”。两者很容易混，但职责不同。

见：[State Estimation](./state-estimation.md)

### 和 TSID / WBC 的关系
如果动力学参数、摩擦、执行器响应差太多，TSID / WBC 的力矩计算就可能变形。

见：[TSID](./tsid.md)

见：[Whole-Body Control](./whole-body-control.md)

### 和 MPC 的关系
MPC 的预测质量高度依赖模型质量。模型错得离谱，预测再漂亮也没用。

见：[Model Predictive Control (MPC)](../methods/model-predictive-control.md)

## 常见误区

### 1. 以为 SysID 做一次就完事
机器人会磨损、载荷会变、温度会变，参数会漂。

### 2. 只辨识刚体参数，不管执行器
很多时候真正搞你的不是质量矩阵，而是 actuator delay 和 friction。

### 3. 离线误差小，不代表闭环一定好
有些参数 fit 数据很好，但放进控制器里未必最稳。

### 4. 激励设计太弱
如果实验动作太单一，很多参数根本不可辨识。

### 5. 把 SysID 和 Domain Randomization 对立起来
这俩不是竞争关系，而是组合拳：
- SysID 给中心
- DR 给鲁棒性

## 在人形机器人里为什么更难

因为人形机器人同时有：
- 高自由度
- 强耦合
- 接触切换
- 执行器复杂
- 足底接触不确定
- 大量未建模柔顺性

所以人形 SysID 往往不能只靠一套“全局完美参数”，而更像：
- 核心参数辨识
- 执行器模型拟合
- 残差补偿
- 域随机化兜底

## 继续深挖入口

如果你想沿着 system identification 继续往下挖，建议从这里进入：

### 论文入口
- [Sim2Real 论文导航](../../references/papers/sim2real.md)
- [Survey Papers](../../references/papers/survey-papers.md)

### 仿真 / 平台入口
- [Simulation](../../references/repos/simulation.md)

## 参考来源

- [sources/papers/system_identification.md](../../sources/papers/system_identification.md) — ingest 档案（Nguyen 2011 / Gautier 激励轨迹 / Hwangbo ActuatorNet 2019）
- Gautier & Khalil, *Direct calculation of minimum set of inertial parameters of serial robots* — 最小参数集辨识经典
- Wensing et al., *Linear Matrix Inequalities for Physically Consistent Inertial Parameter Identification* (2018) — 物理一致性约束辨识
- Hwangbo et al., *Learning Agile and Dynamic Motor Skills for Legged Robots* (2019) — 执行器网络用于模型 gap 处理

## 推荐继续阅读

- Gautier, Khalil, *Direct calculation of minimum set of inertial parameters of serial robots*
- Wensing et al., *Linear Matrix Inequalities for Physically Consistent Inertial Parameter Identification*
- Hwangbo et al., *Control of a Quadrotor With Reinforcement Learning*（可类比理解执行器与模型 gap 处理）

## 一句话记忆

> System Identification 做的是“让模型更像真实机器人”，它是连接控制模型、状态预测和 sim2real 可信度的关键步骤。
