# Ctrl-F: Force as an Interface for Humanoid Loco-Manipulation

- **类型：** paper
- **作者：** Mengjie Zhao, Yuxin Chen, Yiyang Shao, Chuye Hong, Jiakang Chen, Koushil Sreenath, Masayoshi Tomizuka
- **机构：** 加利福尼亚大学伯克利分校
- **论文：** https://arxiv.org/abs/2611.05507
- **原文：** https://arxiv.org/html/2611.05507v1
- **版本：** v1，2026-11-03
- **项目页：** https://forcectrl.github.io/
- **项目页归档：** [Ctrl-F](../sites/ctrl-f.md)
- **核对范围：** 全文 Abstract、§1–6、图1–9、表1–3与参考文献；方法、仿真、硬件与局限均已核对。
- **通读：** 已完成

## 核心摘录

1. 低层无物体训练：按身体 load mask 采样三维力命令并施加反向 wrench，以动作跟踪目标学会抵消载荷；零力起步后逐步扩大采样范围。
2. 高层接冻结低层，以物体/机器人参考、接触位置、物体状态和本体感知生成每接触体三维力。解析物体 wrench 与接触力分配提供在线教师，PPO 叠加逐渐退火的模仿损失；部署不输入质量或测力。
3. 九个仿真任务分手、足、复合三族，每族一个高低层策略对；并非所有任务共用一对，也不支持未见动作直接泛化。
4. 真机 G1 使用 AprilTag 物体位姿；推踢测试加入20 kg载荷，拖拽14 kg成功而20 kg失败；2 kg箱抬起9/10。项目页说明硬件数字为加到物体上的重量。
5. 秤测10/20/30 N命令得到8.8/16.7/24.5 N，不能当精密力跟踪验证；SONIC无载比较有训练动作与机器人配置差异。
6. 截至2026-11-07项目页Code为coming soon，无官方训练/部署仓或独立数据发布入口；浏览器演示不能代替完整复现。

## 对 wiki 的映射

- [Ctrl-F实体](../../wiki/entities/paper-ctrl-f.md)
