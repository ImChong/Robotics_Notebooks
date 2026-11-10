# DiffWBC / OneSteer 项目主页归档

> 来源归档（site）

- **标题（论文正式名称）：** DiffWBC: Unlocking Steerable General Humanoid Motion Tracking via One-Step Diffusion Whole-Body Control
- **项目页当前展示名称：** OneSteer（匿名项目页，名称与 arXiv v1 不一致）
- **类型：** site / project / humanoid-motion-tracking
- **项目页：** https://diffwbc.github.io/
- **论文：** https://arxiv.org/abs/2610.41354
- **入库日期：** 2026-11-10
- **一句话说明：** Unitree G1 通用全身 motion tracking 的单步生成式低层控制器；支持冻结权重的 cost steering、body-group masks、failure scoring。
- **核对范围：** 读取项目页介绍、method、demo、general tracking 对比表、steering 与 failure self-assessment 模块、footer 资源入口；与 arXiv v1 对照名称和 benchmark 口径。
- **通读：** 已完成

## 开源状态

- **未开源（截至 2026-11-10）：** 项目页资源条目显示 “Code (after review)”，无可点击的官方训练/推理/部署源码链接；论文全文开放，但尚不等于可复现源码开放。
- **论文链接展示问题：** 项目页为 “Paper (coming soon)”；正式论文已在 arXiv 发布，真实入口见上文。
- **命名不一致：** 项目页目前写 OneSteer（匿名），arXiv:2610.41354v1 写 DiffWBC；统一按论文标识建立一个实体节点。
- **数值不一致：** 项目页 BONES from-scratch local error 为 22.6 mm，arXiv v1 Table 2 为 20.1 mm，详情页仅采用 v1 评测数值并标注差异。

## 技术与演示核对

- **常规推理：** 50 Hz 时每 tick 一次 generator forward（3.8 ms，RTX 4090）。
- **测试时引导：** 对输入噪声作成本梯度下降；典型两次生成和两次下降，项目页计 20.3 ms；12 步达到 55.1 ms。
- **真机边界：** 演示 Unitree G1 普通 motion tracking；steering 案例（低顶棚、跨栏、手部到达、故障关节、托盘）均标注为 MuJoCo / mjlab 仿真闭环 rollouts。
- **附加地形演示：** 网站还显示地形 token + RL 微调实验，但论文 arXiv v1 未将此列入主要评测；不可把其仿真成果写作论文 v1 已验证的真机成果。

## 对 wiki 的映射

- [DiffWBC 实体](../../wiki/entities/paper-diffwbc.md)
- [论文通读归档](../papers/diffwbc_arxiv_2610_41354_reading.md)
- [补充原文通读归档](../papers/diffwbc_arxiv_2610_41354.md)

## 参考来源

- https://diffwbc.github.io/
- https://arxiv.org/abs/2610.41354
