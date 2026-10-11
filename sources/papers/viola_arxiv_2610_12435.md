# VioLA: Learning Generalist Humanoid Control Policies from Human Data

- **类型：** paper / humanoid control
- **版本：** arXiv:2610.12435v1，2026-10-08
- **作者单位：** Vesoma、ETH Zürich、MPI-IS、University of Tübingen、ELLIS Institute
- **论文：** [arXiv](https://arxiv.org/abs/2610.12435) · [HTML](https://arxiv.org/html/2610.12435)
- **项目页：** [VioLA](https://viola.is.tue.mpg.de/)
- **代码状态：** 论文称代码与 checkpoint 将发布；截至 2026-10-11 未核实到可访问的仓库/权重入口。
- **入库日期：** 2026-10-11

## 核心摘录

1. **跨具身表示：** 人类与机器人动作编码到配对潜空间；策略预测身体、手部运动潜变量，由预训练控制器执行，而非直接预测机器人关节命令。
   - **对 wiki 的映射：** [VioLA 实体](../../wiki/entities/paper-viola-human-data-control.md) 的方法栈。
2. **数据构成：** 训练池 140.6M 帧，其中 93.2% 为人类数据。此比例不是机器人交互数据比例。
   - **对 wiki 的映射：** 实体页数据与适用范围。
3. **真机结果：** 作者报告 G1 locomotion 成功率 100%、manipulation 88.6%，示例含关笔记本、挂衣服、转椅；结果限于论文协议。
   - **对 wiki 的映射：** 实体页评测表与结论。
4. **发布承诺：** 论文称代码与 checkpoints 将发布，入库核查时尚未确认公开入口。
   - **对 wiki 的映射：** 实体页工程实践与局限。

## 原始入口

- arXiv HTML 含摘要、方法、数据与实验。
- 项目页 https://viola.is.tue.mpg.de/ 本次自动访问未返回可读正文，开放状态不据此推断。
