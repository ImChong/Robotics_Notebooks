# WholeBodyWAM（arXiv:2609.16644）

> 来源归档（paper）

- **标题：** WholeBodyWAM: Generalizing Pre-trained World-Action Priors to Humanoid Loco-Manipulation via WBC-Grounded Coordination
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.16644>
- **PDF：** <https://arxiv.org/pdf/2609.16644>
- **项目页：** <https://wholebodywam.github.io/>
- **机构：** 香港中文大学（CUHK）；香港大学（HKU）；北京大学（PKU）；斐研究院（Phi Institute / Φ-Institute）
- **作者：** Zhuo Li, Yiming Yao, Jim Tan, Mengjie Jing, Zhipeng Dong, Fei Chen
- **入库日期：** 2026-09-16
- **一句话说明：** 保留预训练世界—动作先验，用统一 WBC 语义接口与协调感知注意力，将桌面 WAM 泛化到人形 loco-manipulation。

## 开源状态

- **待发布**（步骤 2.5 核查，2026-09-16）：项目页无 GitHub / Hugging Face 链接；BibTeX 为匿名审稿版，作者与发表信息待公开后更新。

## 核心摘录

1. **问题：** 多数 WAM 聚焦桌面或单臂操作；人形 loco-manipulation 需同时协调行走、全身控制与手部动作，不宜从零重学全身行为或简单扩 action 接口。
2. **方法：** WholeBodyWAM 在共享 Diffusion Transformer 内联合预测未来视觉动力学、操作动作与 **UWBC（Unified Whole-Body Controller）** 命令；保留预训练视觉—操作通路，引入结构化动作分解（SAF）与异构 WBC 语义接地。
3. **UWBC 接口：** 56 维槽位（46 共享物理语义 + 10 控制器特定残差）；按注册语义与控制器能力激活字段，不支持项 mask。
4. **CASA 协调：** 当任务方向手臂可操作度下降时，协调门增强 manipulation→UWBC 注意力，让操作意图引导补偿性全身运动。
5. **执行边界：** WholeBodyWAM 预测 intent；下游 WBC 负责平衡、步态、接触与低层跟踪。
6. **评测：** 6 项 SIMPLE 仿真（L0/L1/L2 扰动，SONIC + 每任务 100 条微调示范）+ 8 项真机 loco-manipulation；覆盖 SONIC、AMO、GEAR 三种 WBC。
7. **主要数字（作者报告）：** 仿真总体成功率 **91.9%**（Cosmos-3 **86.4%**）；真机 OOD **68.8%**（DreamZero **40.0%**）；跨 WBC 成功率方差 **10.5 pp²** vs Cosmos-3 **35.0 pp²**（约 **70%** 降幅）；OOD 执行 **81.3%** vs DreamZero **57.5%**。

**对 wiki 的映射**

- [paper-wholebodywam](../../wiki/entities/paper-wholebodywam.md)
- [12 篇技术地图](../../wiki/overview/vla-deploy-12-papers-technology-map.md)
