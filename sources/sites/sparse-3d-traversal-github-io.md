# sparse-3d-traversal-website（Agile Perceptive Traversal 项目页）

- **标题：** Learning Agile Perceptive Traversal of Sparse 3D Structures for Humanoids
- **类型：** site / project-page
- **URL：** <https://nemantor.github.io/sparse-3d-traversal-website/>
- **arXiv：** <https://arxiv.org/abs/2608.29769>
- **入库日期：** 2026-09-02
- **再核日期：** 2026-09-04
- **配套论文：** [Agile Perceptive Traversal（arXiv:2608.29769）](../papers/agile_perceptive_traversal_arxiv_2608_29769.md)

## 一句话摘要

ETH RSL / ETH AI Center / CVG 提出的 **稀疏 3D 结构敏捷感知穿越** 官方站：PM-01 人形 + 头部 RoboSense E1R 固态激光雷达，经 AME-2 注意力编码 + GRU 记忆与分阶段多教师蒸馏，完成 **跳上→荡杆→跳下** 完整猴架序列（15 次真机 14 成功）及 **2 cm 横截面矮身通过**；含方法交互图与真机/仿真注意力可视化。

## 公开信息要点（截至 2026-09-04 再核）

- **机构：** Robotic Systems Lab（ETH Zürich）；ETH AI Center；Computer Vision and Geometry Group（ETH Zürich）。作者：Efe Ongan、Chong Zhang、Boyang Sun、Andrei Cramariuc、Cesar Cadena、Marco Hutter。
- **硬件：** ENGINEAI **PM-01** 人形；被动钩式末端执行器；头部 **RoboSense E1R** 固态 LiDAR（192×144 扫描，策略消费降采样 36×35×4）。
- **方法三模块：** (A) 特权分任务教师 + 阶段调度器（jump-up / brachiation / jump-down）；(B) 感知学生：AME-2 编码器 + GRU + 动作/辅助头；(C) 三阶段蒸馏：DAgger → critic warm-up → 正则化 PPO（BC anchor 衰减）。
- **真机猴架：** 三梯配置共 15 trials，完整序列 **14/15**（93%）；荡杆速度最高 **0.5 m/s**；梯 A 1.69 m / 0.26 m 间距 9/9 全成功。
- **矮身：** 2×2 cm 木条随机朝向，10 次通过 **10/10**。
- **代码 / 数据（步骤 2.5）：** 页面 **无** GitHub / Hugging Face / Zenodo 链接；Footer 仅致谢 EngineAI 与 ETH AI Center 资助。作者账号 [`nemantor`](https://github.com/nemantor) **无** 本项目仓。**无代码入口** → 按 **确认未开源** 处理（2026-09-04 再核与 2026-09-02 初核一致）。

## 关联

- Wiki：[paper-agile-perceptive-traversal-sparse-3d](../../wiki/entities/paper-agile-perceptive-traversal-sparse-3d.md)
- 交叉：[AME-2](../../wiki/entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md)、[LadderMan](../../wiki/entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md)
