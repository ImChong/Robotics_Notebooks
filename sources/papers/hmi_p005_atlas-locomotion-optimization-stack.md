# Optimization-based Locomotion Planning, Estimation, and Control Design for the Atlas Humanoid Robot（Atlas Locomotion，HMI P005）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** Optimization-based Locomotion Planning, Estimation, and Control Design for the Atlas Humanoid Robot
- **短名：** Atlas Locomotion
- **类型：** paper / hmi-papers / 工程与实机部署
- **HMI ID：** P005
- **年份：** 2016
- **原文：** https://doi.org/10.1007/s10514-015-9479-3
- **代码：** 无 / 见正文开源状态
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 把足步/全身规划、状态估计与高频反馈控制放进同一条 Atlas 闭环，强调坐标系、频率与状态接口必须统一。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P005](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P005.md)

## 开源状态（步骤 2.5）

- **结论：** 未开源（工业/实验室闭源系统论文）

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

动态人形系统常见的问题是每个模块单独看都合理：足步规划能给路线，轨迹优化能给全身动作，状态估计能给姿态，控制器能跟踪。但只要坐标系、频率、状态定义或延迟没有统一，整机仍然走不起来。本文的价值正是把规划、估计和控制放进同一条Atlas闭环，而不是只报告一个局部算法。

**对 wiki 的映射：** [`wiki/entities/paper-atlas-locomotion-optimization-stack.md`](../../wiki/entities/paper-atlas-locomotion-optimization-stack.md)

### 摘录 2

系统可以由足步规划器先决定接触位置和时序，再生成行走参考；也可以由全身运动规划器处理更复杂的身体与环境约束。规划结果不是直接发给电机，而是形成名义状态和输入轨迹。控制层围绕这条轨迹使用时变LQR等反馈，根据实时状态偏差修正动作；必要时LQR解可在线重新计算，降低真实机器人偏离名义轨迹后的敏感性。

**对 wiki 的映射：** [`wiki/entities/paper-atlas-locomotion-optimization-stack.md`](../../wiki/entities/paper-atlas-locomotion-optimization-stack.md)

### 摘录 3

状态估计与控制在高频闭环运行，融合机器人传感信息并给控制器提供浮基状态。规划线程和控制线程时间尺度不同：前者可以较慢地解决未来接触和轨迹，后者必须持续吸收估计误差与扰动。系统设计的关键是明确每一层消费什么状态、输出什么参考，以及旧计划在新状态下是否仍然有效。

**对 wiki 的映射：** [`wiki/entities/paper-atlas-locomotion-optimization-stack.md`](../../wiki/entities/paper-atlas-locomotion-optimization-stack.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-atlas-locomotion-optimization-stack.md`](../../wiki/entities/paper-atlas-locomotion-optimization-stack.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
