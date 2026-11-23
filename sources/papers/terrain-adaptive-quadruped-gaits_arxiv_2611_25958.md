# 地形自适应四足最优步态与运动风格（arXiv:2611.25958）

> 来源归档（paper）；本文为原论文的一手材料索引与核对笔记，不存储 PDF 版权原件。

- **标题：** Learning optimal gaits and styles for terrain-adaptive quadruped locomotion
- **中文标题：** 学习地形自适应四足运动的最优步态与运动风格
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.25958>
- **HTML 全文：** <https://arxiv.org/html/2611.25958v1>
- **PDF：** <https://arxiv.org/pdf/2611.25958>
- **版本：** arXiv:2611.25958v1，2026-11-19，预印本
- **作者：** Guillaume Bellegarda；Auke Ijspeert
- **机构：** Inria／Université de Lorraine／CNRS（Nancy）；École Polytechnique Fédérale de Lausanne（EPFL）
- **项目页：** <https://guillaume10.github.io/terrain-gaits/>
- **入库日期：** 2026-11-23
- **核对范围：** 通读 arXiv v1 官方 HTML 全文（引言、Results、Discussion、Methods、Acknowledgements、附录 gait definitions/Stage 1-2 指标/PPO 超参数及训练对照/补充视频说明），对照项目主页 Method、Which gait is best、Zero-shot、Bipedal mode、Hardware 等章节；核对的是论文本身及官网文字，不代表独立复现实验。
- **通读：** 已完成
- **一句话说明：** 用 CPG 耦合振子约束 11 种步态，一阶段学习地形自适应步态参数与姿态，冻结后由二阶段策略按目标指标选择步态，并借助深度图高程估计器部署真机。

## 核心摘录与核验

1. **研究问题：** “最佳步态”不是固定答案，要同时指定地形、评价指标和命令速度。作者系统比较 11 种步态、7 类地形、5 种难度、4 个速度（0.5/1.0/1.5/2.0 m/s）；“最优”仅指在所比较的 11 个预设步态中对给定指标表现最好，5% 内视为并列（Results / Fig. 3）。
2. **两阶段分工：** Stage 1 每几秒随机指定步态的 4×4 相位偏置矩阵，PPO 在相同奖励下学会 11 种步态的振幅、频率和机身高度/摆腿高度/前后足偏移；Stage 2 冻结 Stage 1，只训练输出 11 维选择向量的轻量策略，按能耗、机身角速度或关节加速度分别优化（Methods / Fig. 2）。
3. **物理控制链：** 策略输出 11 维 `[h, g_c, x_off, μ(4), ω(4)]`，CPG 相位耦合 -> 振子状态 -> 足端轨迹 -> IK 关节位置 -> PD 力矩。推理/调制为 33 Hz，振子积分和 PD 为 1 kHz（Methods）。
4. **感知与训练：** 仿真特权输入为 17×11 高程点阵；真机用单个前向 Intel RealSense D435i 的 48×16 深度图，经 CNN + 256 单元 GRU + MLP 预测高程点阵，以 DAgger 训练；Stage 1/2 策略冻结。Isaac Gym/PhysX 训练 4096 个 Go1 并行环境、单张 RTX 3090（Methods）。
5. **主要结果：** 每种步态都在至少一个地形×难度×速度×评价指标组合下进入最优集合；trot 在能耗指标仅约四分之一的测试组合中最优。Stage 2 可在障碍前由步行/小跑转为 bound/gallop（Discussion / Fig. 3–4）。
6. **泛化与真机：** 仿真展示 0.8 m 缝隙（训练最大 0.55 m）、0.55 m 墙、0.4 m 台阶，以及 Go1 到 Go2/Aliengo/B2（B2 约训练质量 5 倍）零样本迁移；Go1/Go2 真机用机载深度估计通过 0.7 m 缝隙和 0.21 m 障碍。另增加 pitch 跟踪奖励可训练前腿/后腿双足模式（Results / Supplementary Videos）。
7. **局限：** 强耦合限定相位而非保证实际落足接触，宽缝隙上接触时序仍会偏离；mixed blocks 偏心碰撞和横向运动能力不足；Stage 2 改目标不联合重新优化 Stage 1 风格，尚未验证泥雪等可变形地形或根本不同的腿机构（Discussion）。

## 开源状态核查

- 2026-11-23 实际打开官方[项目页](https://guillaume10.github.io/terrain-gaits/)：展示论文、视频和“Code”文本，但网页提取内容没有给出可核实的 GitHub/代码下载地址；arXiv v1 亦未提供可运行仓库 URL。
- **结论：官方可运行代码入口未确认**；未读取训练/部署源码，不能写源码运行时序图，也不把作者其他 CPG 仓库冒充本论文代码。
- 项目页来源归档：[terrain-adaptive-quadruped-gaits.md](../sites/terrain-adaptive-quadruped-gaits.md)。

## 对 wiki 的映射

- [独立论文节点：地形自适应四足最优步态与风格](../../wiki/entities/paper-terrain-adaptive-quadruped-gaits.md)
- 对应 [感知越障纵深路线](../../roadmap/depth-perceptive-locomotion.md) Stage 3：地形感知之后的多步态选择与跳跃越障。
