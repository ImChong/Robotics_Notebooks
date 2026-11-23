# DemoGrasp: Universal Dexterous Grasping from a Single Demonstration

> 来源归档（paper / arXiv:2509.22149v1，ICLR 2026）

- **URL：** <https://arxiv.org/abs/2509.22149>
- **PDF：** <https://arxiv.org/pdf/2509.22149.pdf>
- **项目页：** <https://beingbeyond.github.io/DemoGrasp/>
- **代码仓库：** <https://github.com/BeingBeyond/DemoGrasp>
- **标题：** DemoGrasp: Universal Dexterous Grasping from a Single Demonstration
- **作者：** Haoqi Yuan, Ziye Huang, Ye Wang, Chuan Mao, Chaoyi Xu, Zongqing Lu
- **机构：** BeingBeyond（智在无界）、Peking University、Renmin University of China
- **年份：** 2025/2026（arXiv 2025-09；ICLR 2026）
- **核对范围：** 已获取并通读 `arXiv:2509.22149v1` PDF 抽取全文（Sections 1–5 及附录，涵盖单步 MDP RL 编辑机制、7000 并行 IsaacGym 环境、35K 成功轨迹蒸馏 Flow-Matching 视觉策略、DexGraspNet 与 110 件真机未知物体评测）；未独立复现仿真与真机实验。
- **通读：** 已完成

## 摘要与核心贡献

1. **单演示 + 单步 MDP 强化学习泛化**：仅用 1 条人工（或遥操作）灵巧抓取参考姿势与轨迹，将抓取泛化重构为**一步马尔可夫决策过程（1-step MDP）**：RL 策略仅预测手腕相对物体中心的 $\text{SE}(3)$ 残差位姿 $T^{ee}$ 与关节增量 $\Delta q^G$，随后交由**确定性开环轨迹重放**（趋近 -> 合拢 -> 抬升），绕开长时域接触奖励稀疏难题。
2. **极简二值碰撞惩罚奖励与高效仿真采样**：使用 `R = 1` 当且仅当成功抬起物体且**非预设接触区无碰撞**；为适应薄/扁平物体（需贴近桌面抓取），在训练期为 **50% 并行环境关闭桌面自碰撞检测**。在 7,000 个 IsaacGym 并行环境中高效探索生成 **35K 条成功抓取轨迹**。
3. **闭环视觉 Flow-Matching 蒸馏与 Sim2Real**：将成功轨迹蒸馏到基于单目多视角 / ViT 的 **Flow-Matching** 视觉闭环策略（Action Chunking），直接消除真机位姿估计误差。
4. **实验表现**：
   - DexGraspNet（Shadow Hand，全物体集）：状态策略在 train / seen / unseen 分别达 **95.2% / 95.5% / 94.4%**，视觉策略达 **92.2% / 92.3% / 90.1%**；
   - 真机 Franka FR3 + Inspire Hand（110 件未见日常物体）：总成功率 **86.5%**（正常尺寸 **95.3%**，极小/极薄物体 **71.1%**），杂乱多物体场景达 **82%–84%**。

## 与知识库关联

- 实体节点：[cn-os-demograsp.md](../../wiki/entities/cn-os-demograsp.md)
- 代码归档：[demograsp.md](../repos/demograsp.md)
