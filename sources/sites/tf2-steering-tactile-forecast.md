# TF²-Steering（作者官方项目页）

> 来源归档（site）；与论文节点共用项目身份，**不新建重复 wiki 实体**。

- **标题：** TF²-Steering: Action Refinement with Tactile Forecasts and Feedback via Velocity-Field Steering
- **类型：** site / author project page
- **链接：** <https://yiming-cc.github.io/tf2-steering/>
- **论文：** <https://arxiv.org/abs/2611.01483>
- **全文 HTML：** <https://arxiv.org/html/2611.01483>
- **初次入库：** 2026-11-04
- **复核日期：** 2026-11-10
- **核对范围：** 项目页页首标题、作者/机构、PDF/arXiv/Code 栏、Tasks、Method、Results 与页面底部；对比论文第 3–4 节。
- **通读：** 已完成
- **对应实体：** [TF²-Steering](../../wiki/entities/paper-tf2-steering-tactile-forecast.md)

## 源码 / 模型 / 数据开放状态

截至 2026-11-10，官方页面有 “Code” 文本，但未提供可点击的 GitHub 源码 URL；页面可查看方法摘要与六项真机任务演示，没有核实到本项目可下载权重或训练数据集链接。**开源状态：待发布 / 暂无官方可运行实现入口**。不将论文中引用的 T-REX、FTP-1 或 LTX-Video 仓库误认作 TF²-Steering 的官方实现；不绘制臆造的源码调用时序图。

## 页面与正文交叉核对

- 项目页的“Predict once. Refine as you act”对应论文 §3.2：chunk 期间缓存世界模型先验，再用新到达的触觉反馈修正尚未执行动作。
- 项目页宣称六项仿真与六项真机评测，对应原文 Tables 1–2；精确数字、对照与 PSR 定义统一以 [论文归档](../papers/tf2_steering_tactile_forecast_arxiv_2611_01483.md) 为准。
