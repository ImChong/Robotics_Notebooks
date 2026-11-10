# LingBot-Touch — 技术报告原文核验记录（待完成）

- **英文标题：** Learning Heterogeneous Tactile Representation for General Dexterous Manipulation
- **项目 / 作者：** Robbyant（蚂蚁灵波）；GitHub README 列 Bing Han、Yichuan Peng、Haoyu Dong 等作者，通讯作者 Wei Ning
- **PDF 一手原文：** <https://github.com/Robbyant/lingbot-touch/blob/main/paper/Learning_Heterogeneous_Tactile_Representation_for_General_Dexterous_Manipulation.pdf>
- **项目主页：** <https://technology.robbyant.com/lingbot-touch>
- **代码：** <https://github.com/Robbyant/lingbot-touch>
- **核验日：** 2026-11-10
- **PDF 文件身份：** Git LFS 文件，指针给出 SHA-256 `1d68b6ce43470b9ffbdfd7425480581845b2515eb7897661e26152c5b257a3ce`、大小 19,010,130 bytes。
- **核对范围：** GitHub `paper/` 目录及 Git LFS 指针；尝试官方 PDF GitHub Raw / media.githubusercontent 读取但接口无法获取二进制正文；论文 `1–N、表格、附录、具体损失函数及消融尚未逐页核对。已独立阅读官方项目主页、官方 README、数据接口指南及公开推理入口（分别归档在 sites/repos）。
- **通读：** 未完成

## 已由其他一手来源证实的背景（不能冒充论文全文核验）

- 官方网页披露 4,983 小时、464,431 episodes、约 1.33B 帧，来自压阻、磁性、视触觉三类传感器。
- 官方 README 描述按模态 tokenization、共享 Haptic Transformer、掩码潜空间预测的 JEPA 预训练；推理发行包只包含冻结 target encoder。
- 官网给出 26 个真实灵巧操作任务（Sharpa Wave 20、XHand 6，每项 20 次）及 12 个模拟强化学习任务（每任务 3 seeds）。
- 以上事实来自官网/README，不是对论文 PDF 具体表格逐页复核；独立实验复现没有进行。

## 待完成的全文核验

1. 下载并通读 19 MB 技术报告；核实预训练 JEPA 损失、时空地址编码、teacher/student EMA 参数及训练协议。
2. 对照论文表格提取 26 真机任务、四类 policy、12 仿真任务的完整基线、置信区间/种子与消融；不要从网页图形推断不可读数字。
3. 交叉验证训练数据是否公开、Tiny/Small 参数规模与速度测量硬件、跨硬件迁移条件和限制。
4. 完成后将本文件改成 `通读：已完成`，同步审阅 wiki 的 `status`、PR 健康度，并将 Draft PR 转为 Ready。

## 对 Wiki 的映射

- [LingBot-Touch 主实体](../../wiki/entities/paper-lingbot-touch.md)
- [官方项目页核验](../sites/lingbot-touch-official.md)
- [官方推理源码核验](../repos/lingbot-touch-official.md)
