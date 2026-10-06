# Gated Residual Body–Hand Coordination for Whole-Body Humanoid Teleoperation

> 来源归档（paper）；主来源为作者上传至 arXiv 的 v1 论文 HTML/PDF。

- **标题：** Gated Residual Body–Hand Coordination for Whole-Body Humanoid Teleoperation
- **作者：** Ruiming Wu, Shuang Li, Liding Zhang, Alois Knoll, Zhaopeng Chen
- **单位：** Technical University of Munich（TUM）与 Agile Robots SE；各作者署名见 arXiv 原文
- **类型/状态：** arXiv 预印本，cs.RO，v1，2026-09-16
- **arXiv：** <https://arxiv.org/abs/2609.18763>
- **HTML：** <https://arxiv.org/html/2609.18763>
- **PDF：** <https://arxiv.org/pdf/2609.18763>
- **入库复核：** 2026-10-06
- **一句话说明：** 冻结全身跟踪器与手部重定向器，仅训练带动作条件权限门控和人体参考几何奖励门控的有界身手残差；held-out 仿真 GRAB 上腕/指尖几何误差降低 39.2%–56.3%。

## 公开实现状态

截至 2026-10-06，论文首页与正文中未发现作者项目页或本文专属代码仓库链接，故记录为**未发现公开发布**。论文说明使用 NVIDIA SOMA Retargeter 与 DexPilot 风格手部重定向等组件；这些依赖不等于本文协调策略的开源实现。评测为仿真，单 Agile One 具身、单训练随机种子。

## 核心摘录

1. **问题：** 独立的全身动作跟踪器与灵巧手重定向器直接组合，不会自动保持两路命令之间的腕部、双手与指尖几何关系。
2. **形态标定：** 从 7 组人工配对人–机器人姿态联合估计根坐标系三轴尺度与末端局部偏移；配合 SOMA Retargeter 对 52,663 对候选动作重定向。
3. **语料整理：** 几何过滤、质量筛选、瞬时 IK 跳变修复、足底对齐、密度感知筛选后得到 20,000 条、43.67 小时的全身参考。
4. **名义控制器：** SONIC-based 跟踪器在上述语料上经 30,000 次 PPO 迭代训练（8×RTX PRO 6000，约 1.1k GPU 小时）后冻结；DexPilot 风格模块生成双手共 32 维手指关节目标。
5. **残差接口：** 共享 actor 主干分别预测 29 维身体与 32 维手部修正；tanh 与固定尺度限制修正幅度，初始零输出等价于原始直接组合。
6. **动作门控：** 用人体参考时间窗内的腕/指活动、双腕接近度、腿部范围和脚步特征，在执行时调节手/上身与腿部残差权限；使用名义系统已有 180 ms 参考缓存。
7. **奖励门控：** 训练期根据人体参考的双指尖距离、双腕距离与腕部方向，改变交互几何奖励的权重；不依赖显式物体状态或接触标签。
8. **残差训练：** 2,304 段 GRAB（含镜像）和 1,006 段 PICO 参考，PPO 5,000 次迭代，8×RTX PRO 6000。
9. **主要评测：** 50 段 held-out GRAB，包含 10 位受试者、26 类物体、15 种意图和五种动作情境；每段 10 个匹配副本。所有消融设置的根跟踪存活成功率均为 100%，故作者以连续几何误差比较协调能力。
10. **结果：** 相对直接组合，完整残差腕间距离误差 10.26→4.62 cm、腕旋转误差 29.15→17.73°、双手指尖间误差 11.61→5.07 cm、拇指–指尖误差 4.33→2.28 cm；对应下降约 39.2%–56.3%。
11. **跟踪保持：** 同一关节手模型下 AMASS 存活率 89.03%→89.29%；根部朝向误差略降，但根部 XY、身体位置与身体朝向误差略升，不能概括为各项跟踪均改善。
12. **消融观察：** 常数奖励门控具有最低聚合协调误差；完整模型在近距离双手窗口更优，说明方法存在窗口性几何收益与总体指标之间的取舍。

**对 wiki 的映射**

- [paper-gated-residual-body-hand-coordination](../../wiki/entities/paper-gated-residual-body-hand-coordination.md)
- [周更盘点来源](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md)

## 一手来源

- [arXiv abstract/version record](https://arxiv.org/abs/2609.18763)
- [arXiv HTML 全文](https://arxiv.org/html/2609.18763)
- [arXiv PDF](https://arxiv.org/pdf/2609.18763)
