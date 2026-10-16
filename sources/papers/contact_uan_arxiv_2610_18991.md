# Contact-UAN: Unsupervised Actuator Networks as a Residual World Model for Humanoid Control

> 来源归档（paper / arXiv v1）

- **作者：** Nolan Fey、Khai Nguyen、Yash Prabhu、Gabriel B. Margolis、Pulkit Agrawal
- **机构：** 麻省理工学院 Improbable AI Lab
- **提交日期：** 2026-10-13（arXiv:2610.18991v1）；截至 2026-10-16 收录 arXiv 预印本。项目页另注明获 CoRL 2026 接收。
- **论文：** <https://arxiv.org/abs/2610.18991>
- **全文：** <https://arxiv.org/html/2610.18991v1>
- **项目页：** <https://contact-uan.csail.mit.edu/>
- **官方代码：** <https://github.com/nolie-rolie/contact-uan>
- **源码归档：** [项目页核查](../sites/contact-uan-csail-mit-edu.md)；[仓库核查](../repos/contact-uan.md)
- **沉淀到 wiki：** [Contact-UAN 实体页](../../wiki/entities/paper-contact-uan.md)

## 摘要式定位

Contact-UAN 面向人形机器人接触运动中的执行器 sim-to-real 缺口：保留仿真刚体与接触求解器，只学习对关节位置目标的残差修正。关键前提是先用本体感觉估计足部接触姿态，给硬件轨迹重放设置接触一致的仿真初态；否则残差网络会把重置误差当成执行器误差学习。

## 方法与数据要点

- **接触一致状态估计：** 短时序 IMU、本体关节状态和上一时刻关节目标用于预测接触概率、支撑脚仿真姿态及基座速度；只对置信度高的支撑脚回归穿透高度与倾角，再融合候选根姿态。估计器只用于离线重标注，不进入部署策略。
- **UAN 残差：** 网络输出关节目标修正 `δq`，使仿真重放的关节转移更接近硬件；基准仿真仍负责刚体动力学、接触和积分。每种同类执行器共享两层 MLP，观测含 3 步目标误差与关节速度历史（60 ms）。
- **采集与训练：** 在 Unitree G1 上用 WTW 风格步行策略采集约 6.5 分钟本体数据，改变步频和摆腿高度以覆盖不同支撑、摆动和冲击。论文名义仿真用 BeyondMimic G1 参数；UAN 重放任务使用 4096 个并行环境和 24 步窗口。
- **策略训练与部署：** 冻结 UAN 后，在增广仿真中训练下游跑步、力控和跳跃策略；UAN 不随任务策略部署到真机。

## 主要结果

| 评测 | 论文报告 | 读法 |
|------|----------|------|
| 真实机器人 | 29 自由度 Unitree G1；单一 UAN 从步行数据迁移到跑步、力控、阻抗和跳跃 | 支持跨多类执行器敏感任务的迁移证据来自同一平台 |
| 跑步 | 相近速度跟踪下，运行扭矩用量降低约 20%；3.0 m/s 指令下达到 3.13 ± 0.06 m/s | 改善的是效率与执行可行性，不只是运动外观 |
| 力控 | 平均力跟踪误差降低 42.8 ± 12.1% | 对接触力和全身承载敏感的场景最相关 |
| 仿真重放 | 接触一致重置相对直接估计使困难 sim-to-sim 设置的 20 秒存活率由 49.8% 升至 96.7%；UAN 将硬件扭矩总量不匹配降低 69.2% | 初始接触状态校准和执行器残差各自重要 |

真机跑步每种策略/速度做 5 次试验，力控各条件做 4 次；结果仍需在更多硬件、执行器温度与任务上复核。

## 证据边界

- 论文方法依赖一个可用的刚体/接触仿真先验；它不替代整套动力学建模，也不从数据中学习完整世界模型。
- 当前模型按执行器类型共享参数，但未建模执行器间耦合或电机温度动态；短时数据对缓慢热效应尤其不足。
- 公开仓库含训练、估计、重放与部署参考代码，Apache-2.0；真机数据、预训练策略不随仓库发布，复现实验需要自有 G1 和数据采集。

## 关联资料

- [Contact-UAN 项目页核查](../sites/contact-uan-csail-mit-edu.md)
- [Contact-UAN 官方源码核查](../repos/contact-uan.md)
- [Contact-UAN Wiki 实体页](../../wiki/entities/paper-contact-uan.md)
- 相关方法：[执行器网络](../../wiki/methods/actuator-network.md)、[仿真到真实迁移](../../wiki/concepts/sim2real.md)
