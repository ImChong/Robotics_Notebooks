# cbschaff/rsa（Residual Shared Autonomy 官方实现）

> 来源归档

- **标题：** Residual Policy Learning for Shared Autonomy
- **类型：** repo
- **来源：** TTIC（Charles Schaff）
- **链接：** <https://github.com/cbschaff/rsa>
- **项目页：** <https://ttic.uchicago.edu/~cbschaff/rsa/> — 归档见 [`sources/sites/rsa-ttic.md`](../sites/rsa-ttic.md)
- **入库日期：** 2026-07-28
- **一句话说明：** RSA（ICRA 2020）官方代码：约束 PPO（Lagrangian 残差幅值正则）训练的共享自治 copilot，含 Lunar Lander / Lunar Reacher / Drone Reacher 三环境与代理人类 pilot（BC 模仿）训练流程。
- **沉淀到 wiki：** [`wiki/entities/paper-residual-policy-shared-autonomy.md`](../../wiki/entities/paper-residual-policy-shared-autonomy.md)

---

## 核心定位

论文配套仓库（GitHub 描述："Code for the paper Residual Policy Learning for Shared Autonomy"）。实现要点与论文对应：

| 组件 | 说明 |
|------|------|
| Copilot | 三层 MLP（128 hidden），policy/value 双头；约束 PPO + softplus Lagrange 乘子；残差 $a_r\sim\pi_r(s,a_h)$ 与人动作 $a_h$ 相加执行 |
| Surrogate pilots | 对 9（Lander）/ 14（Drone）名参与者各自 BC 模仿，训练时按概率 0.001 逐步切换；100K 步 value warm-up（copilot 输出 0） |
| 环境 | Lunar Lander / Lunar Reacher（OpenAI Gym 改造）、Drone Reacher（6-DoF 四旋翼，15 维状态 + 随机风扰） |

## 运行要点

- 训练规模：论文设置 100M timesteps，每 20M 衰减学习率（×√0.1）；评估每个 pilot–copilot 组合 1000 episodes。
- 复现注意：人测数据不随仓发布；BC 代理 pilot 需先按仓库流程自行采集/训练。

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-residual-policy-shared-autonomy](../../wiki/entities/paper-residual-policy-shared-autonomy.md) | 本仓库对应的论文实体页 |
| [paper-residual-policy-learning](../../wiki/entities/paper-residual-policy-learning.md) | 方法源头（RPL）；RSA 把 base policy 从控制器替换为人 |
