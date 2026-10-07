# InterMimicGen: Scaling Humanoid Loco-Manipulation through Self-Evolving Motion Imitation

> 来源归档（ingest · arXiv 预印本）

- **类型：** paper / humanoid / loco-manipulation / motion-retargeting / imitation-learning
- **arXiv：** <https://arxiv.org/abs/2610.06850>（v1，提交于 2026-10-05；HTML：<https://arxiv.org/html/2610.06850v1>）
- **项目页：** <https://sirui-xu.github.io/InterMimicGen/>
- **作者：** Yucheng Zhang、Sirui Xu、Jinhong Li、Liuyu Bian、Anatulya Nandi、Derek Zhang、Xiangchen Liu、Xueting Li、Umar Iqbal、Yu-Xiong Wang、Liang-Yan Gui
- **机构：** 伊利诺伊大学厄巴纳-香槟分校（UIUC）；英伟达（NVIDIA）
- **开源状态：** 截至 2026-10-07，项目页只列论文与研究演示，没有链接此项目的算法源码、数据或模型仓库。相关的 MimicGen / HumanoidMimicGen 不等于本项目实现。
- **对 wiki 的映射：** [InterMimicGen](../../wiki/entities/paper-intermimicgen.md) — 单项目实体节点，包含论文与项目页，不另建 paper/project/repo 节点。

## 问题与主张

人体–物体交互捕捉数据规模有限、来源异质，且参考动作不能直接由机器人动力学执行。InterMimicGen 把交互数据整理与人形重定向、物理跟踪和自演进扩增串成一个闭环：跟踪器校正/验证动作，验证通过的变体再成为下一轮训练数据。

## 方法摘录

1. **融合参考数据：** 汇集 InterAct 与 HiPHI 的交互动作，统一人体、物体和手部表示；依目标机器人的运动能力筛除不可支持的片段。
2. **保留接触的重定向：** 先通过交互网格进行全身身体–物体关系求解，并加入拇指与其他手指对向的抓握约束；再做两轮几何清理，降低穿透、抖动和接触漂移。
3. **训练通用跟踪器：** 基于 PPO 在并行物理仿真中训练一个共享策略，观测机器人状态、未来参考帧与物体几何关系，输出 PD 关节目标；奖励同时约束身体、物体、交互几何和接触。
4. **自演进数据飞轮：** 对已验证的父动作做两类小幅编辑：移动/旋转物体交互路径，或改变完成动作的身体姿势。保持交互类型、意图接触、物体和任务结果等语义不变；每轮先在候选数据上微调跟踪器，再在物理仿真里执行，只保留完成任务且动作质量合格的轨迹作为下一轮种子。

## 实验摘录

- 参考集合：InterAct + HiPHI 共 **16,059 段动作、140.68 小时、157 个对象条目**，重定向至 6 种机器人配置：G1（普通手、Inspire 或 Dex3）、Booster T1、Booster K1、Dexmate Vega。
- **重定向（G1 + Inspire，3,059 个 OMOMO 片段）：** 相较 Weave，方法的帧级手接触保持率为 98.2% vs 96.1%，物体穿透帧 37.2% vs 67.4%，脚滑帧 12.3% vs 62.0%，身体 MPJPE 5.74 cm vs 9.54 cm；手内滑移略高（0.239 vs 0.225 m/s）。
- **通用跟踪：** 一个策略覆盖 15 种对象。在 G1 + Inspire 上，bimanual / sitting / grasping 成功率分别为 73.2% / 64.9% / 58.3%；每对象专用策略为 93.8% / 75.4% / 84.9%。共享策略的身体误差接近专用策略，但成功率和物体旋转误差仍有差距。
- **五轮演进：** 验证动作集增长至原种子的约 **142–151 倍**；微调后新动作成功率从原跟踪器的 52.1–64.2% 提升至 98.4–98.9%，且原始动作仍保持 100% 成功。脚滑与身体加速度随更远变体有所上升。
- **真机迁移：** 展示 G1 拉行李箱、G1+Inspire 搬三脚架、Booster K1 搬箱、Dexmate Vega 移动椅子。控制细节依平台而异；例如 G1 参考输入 50 Hz、关节 PD 500 Hz。

## 证据边界

- 结果来自 arXiv v1；项目页提供演示材料。动作集、训练权重与算法代码没有由项目页公开链接，不能据论文结果推断读者可直接复现完整管线。
- 「动作集增长 100 倍以上」针对实验中的交互语义和演进设置，不代表新增任务类别；论文明确指出该方法扩展的是已有演示附近的可执行变化。
- 仿真筛选与策略微调对增长都关键。冻结跟踪器在单独消融中第二轮起候选通过率降至 0.4%，增长基本停止。
- 论文承认迭代筛选可能偏向更容易的物体，也可能累积参考误差。

## 对 wiki 的映射

- 主实体：[wiki/entities/paper-intermimicgen.md](../../wiki/entities/paper-intermimicgen.md)
- 相关工作：[HumanoidMimicGen](../../wiki/entities/paper-humanoidmimicgen.md)；两项工作均扩增人形移动操作数据，但数据来源与闭环机制不同。

## 核查记录

- 论文与作者机构依据 arXiv 摘要 / HTML；实验数字依据 arXiv v1 §4 与附录。
- 项目页核查日期：2026-10-07。检查了项目页导航、arXiv 条目与公开 GitHub 搜索结果，没有找到本项目源码入口；未把 MimicGen 上游或相关低层控制仓库冒充为 InterMimicGen 官方代码。
