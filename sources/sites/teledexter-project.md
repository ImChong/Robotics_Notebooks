# TeleDexter 项目页

> 来源归档（site / project page）

- **标题：** TeleDexter: Towards Human-level Dexterous Teleoperation
- **类型：** project page / blog
- **URL：** <https://bigai-dex.github.io/blog/teledexter/>
- **论文：** <https://arxiv.org/abs/2607.11481>
- **PDF：** <https://bigai-dex.github.io/blog/teledexter/paper_teledexter.pdf>
- **代码：** 截至 **2026-07-28** 项目页 metalinks 仅列 arXiv，**未列 GitHub / Hugging Face / 数据集**
- **机构：** BIGAI Dexterity Team（清华 / BIGAI / 北大）
- **发布日期：** 2026-07-12（项目页侧栏）
- **核查日期：** 2026-07-28
- **一句话说明：** BIGAI 灵巧手「小脑」式 hand–object co-tracking 遥操作项目页：75.2% 七任务平均成功率、多阶段工具使用与手内重定向演示，并展示从遥操作到 Diffusion Policy 自主策略的数据飞轮叙事。

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 项目页 Code / Resources | **无** — metalinks 仅 `Paper → arXiv` |
| GitHub 搜索 `teledexter` | **0** 公开仓库（2026-07-28） |
| `bigai-dex` org | API 返回 404 / 无公开仓列表 |
| 论文 PDF Code availability | **未承诺具体仓库 URL**；训练叙述用 Isaac Gym + SAPG |
| 综合判定 | **未开源**（截至入库日） |

## 核心摘录（归纳，非全文）

- 定位：learned cerebellum — 低层 co-tracking 控制器把操作员意图映射为接触丰富的实时执行；仿真训练、零样本真机部署。
- 关键数字：七项灵巧遥操作任务平均 **75.2% SR / 87.1% TP**；基线 DexRT / GeoRT / DexGen 接近失败。
- 任务谱：Hammer / Brush / Screwdriver / Bulb 多阶段工具使用；Cylinder / Cuboid / Bunny 手内重定向；LeapHand 跨具身。
- 数据引擎：每任务约 **50** 条示范可训 RGB Diffusion Policy（BulbInstall / HammerDriver / BrushForward）。
- 局限（项目页）：object-specific 控制器；依赖重型 MoCap；下一步为 object-general + 视觉追踪。

## 对 wiki 的映射

- [TeleDexter 论文实体](../../wiki/entities/paper-teledexter.md)
- [Teleoperation 任务页](../../wiki/tasks/teleoperation.md)
- [深度遥操作路线 Stage 4](../../roadmap/depth-teleoperation.md)

## 参考来源（原始）

- 项目页：<https://bigai-dex.github.io/blog/teledexter/>
- arXiv：<https://arxiv.org/abs/2607.11481>
- PDF：<https://bigai-dex.github.io/blog/teledexter/paper_teledexter.pdf>
