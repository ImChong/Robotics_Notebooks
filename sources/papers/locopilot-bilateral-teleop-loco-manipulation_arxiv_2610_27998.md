# LocoPilot（arXiv:2610.27998）

> 来源归档（paper）

- **标题：** LocoPilot: Single-Operator Bilateral Teleoperation for Loco-Manipulation Policy Learning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.27998>
- **PDF：** <https://arxiv.org/pdf/2610.27998>
- **作者：** Antonio Langella, Giovanni Minelli, Giovanni B. Dessy, Lorenzo Amatucci, Giulio Turrisi, Victor Barasuol, Claudio Semini
- **arXiv 评论栏：** （无）
- **外链：** <https://github.com/iit-DLSLab/LocoPilot>
- **入库日期：** 2026-10-26
- **一句话说明：** LocoPilot：面向移动操作策略学习的单人双向遥操作。

## 开源状态

- arXiv 摘要页/评论给出 GitHub 链接（https://github.com/iit-DLSLab/LocoPilot；仅核对到链接存在，未核验仓库内容）（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】人形／四足39篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
2. **周报所述问题：** 单人难以同时控制四足底盘、身体姿态、机械臂和夹爪，而无实机示范又缺少平台特有观测和动作。
3. **周报所述创新点：** 用主机械臂与两个手柄实现全系统同步控制，以无力传感器方式估计并反馈接触力；数据采集和策略部署共享同一个 11 维动作接口，避免额外动作重定向。
4. **arXiv 摘要（英文节选）：** Learning loco-manipulation policies requires demonstrations that capture coordinated base and arm motion on the target embodiment. Robot-free interfaces allow demonstrations to be collected independently of the target robot, increasing flexibility but without directly capturing platform-specific commands, proprioception, and body-mounted observations. On-robot teleoperation allows to capture these signals directly, but requires an interface capable of commanding the complete system concurrently.

**对 wiki 的映射**

- [paper-locopilot-bilateral-teleop-loco-manipulation](../../wiki/entities/paper-locopilot-bilateral-teleop-loco-manipulation.md)
