# Humanoid Hanoi: Investigating Shared Whole-Body Control for Skill-Based Box Rearrangement

> 论文一手资料核对归档（复用既有唯一论文节点，不存储受版权保护的全文）

- **类型：** paper / CoRL 2026（官方主页标明 accepted）
- **论文：** <https://arxiv.org/html/2602.13850>
- **arXiv ID / 版本：** 2602.13850v4，2026-10-11
- **PDF：** <https://arxiv.org/pdf/2602.13850>
- **项目：** <https://osudrl.github.io/Humanoid_Hanoi/>
- **代码：** <https://github.com/osudrl/Humanoid_Hanoi>（仅基准环境，无训练代码）
- **作者：** Minku Kim、Kuan-Chia Chen、Aayam Shrestha、Li Fuxin、Stefan Lee、Alan Fern
- **机构：** 俄勒冈州立大学（Oregon State University）
- **核对范围：** arXiv v4 HTML 全文 §1–8、Figures 1–6、Tables 1、附录 A（MHC 掩码及数据聚合）、B（四技能、PPO 与 DR）、C（仿真分布与误差/失败模式）、D（真机单技能比较）、E（真机硬件设置、感知、失败和精简两塔实验）；官方项目页与公开源码 README/入口也已核对。前版笔记“实验与附录略读”记录由本次 v4 完整核对更新。
- **通读：** 已完成
- **复核日期：** 2026-11-22

## 核心摘录（归纳，非全文）

1. **问题定义（§3）：** 三箱、三塔、七次移动，每次包含 GoTo → Pickup → GoTo-with-Box → Place；仿真对照均使用相同 100 个随机实例，含 DR / 无 DR 两种环境。
2. **系统（§4，附录 A.1/B）：** 高层四种 LSTM 技能以 50 Hz 发 masked directives；共享预训练 MHC 输出 20 关节 PD 位置目标，下层 PD 2 kHz 跟踪。PDDL 规划接口有实现，但实验采用汉诺塔递归、未评测规划性能。
3. **扩展方式（§5，附录 A.2）：** Pickup/Place 各 9 条成功 directive+mask 序列，采集时**无**动力学随机化；与原参考集拼接，**继续训练时**使用原跟踪目标、DR 与扰动。明确不称为新的 DAgger 算法。
4. **仿真（§6，Fig.4）：** DR 条件下 Base 24%、Finetune 21%、Residual 19%、Joint 16%、Extended **47%** 完整完成率；Extended 单技能全高度 DR Pickup **96.92%**、Place **93.03%**（Table 1）。
5. **控制器交接干预（§6.4）：** 10 步线性插值使 Finetune 21→32%、Residual 19→32%、Joint 16→38%；仍低于 Extended 47%，说明过渡冲击只是部分原因。
6. **真机（§7/附录 E）：** Digit V3、两台 D455 + AprilTag/IMU，三种配置各 5 次，Extended 3/5、2/5、1/5，总计 **6/15**；其他基线 **0/15**。成功整段约 5 分钟，精简两塔任务连续成功五次但并非三塔完整任务。
7. **边界（§8、附录 C.3/C.4/E）：** 以状态式策略研究控制及组合，仿真不是视觉策略；长时程多数故障发生在 Place；不同成功子集的放置误差不能直接作跨方法精度排行。真机感知丢标记、偏航误差与机械碰箱属于单独限制。

## 对 wiki 的映射

- [唯一论文详情节点](../../wiki/entities/paper-notebook-humanoid-hanoi-investigating-shared-whole-body-c.md)
- [移动操作 Stage 1 路线](../../roadmap/depth-loco-manipulation.md)
- [官方项目归档](../sites/humanoid-hanoi.md)
- [官方源码归档](../repos/humanoid-hanoi.md)

## 原始来源

- [论文原文 HTML v4](https://arxiv.org/html/2602.13850)
- [官方项目页](https://osudrl.github.io/Humanoid_Hanoi/)
- [官方代码（环境）](https://github.com/osudrl/Humanoid_Hanoi)
