# Contact-UAN 官方项目页核查

> 来源归档（site / MIT CSAIL）

- **URL：** <https://contact-uan.csail.mit.edu/>
- **标题：** Contact-UAN: Unsupervised Actuator Networks as a Residual World Model for Humanoid Control
- **核查日期：** 2026-10-16
- **论文：** <https://arxiv.org/abs/2610.18991>
- **代码：** <https://github.com/nolie-rolie/contact-uan>
- **数据/权重：** 项目页未提供公开下载链接；仓库说明真机数据和预训练策略不包含在代码发布中。
- **开放状态：** 代码已公开，Apache-2.0；复现所需的真机采集数据和预训练策略需自行准备。
- **源码归档：** [官方仓库核查](../repos/contact-uan.md)
- **论文归档：** [arXiv:2610.18991](../papers/contact_uan_arxiv_2610_18991.md)
- **Wiki 主实体：** [Contact-UAN](../../wiki/entities/paper-contact-uan.md)

## 项目页内容

项目页将方法概括为：先以本体感觉推断足底接触一致的仿真重置，再训练残差 UAN，最后冻结 UAN 训练下游策略。作者报告仅用 6.5 分钟步行数据训练的 UAN，可支持跑步、力控制、阻抗控制和动态跳跃；页面展示的代表指标包括扭矩用量降低约 20%、力跟踪误差降低约 43%。

## 开源核查

项目页提供论文和官方 GitHub 链接。GitHub README 给出从 WTW 策略训练、G1 数据采集、状态估计器训练、数据重标注、UAN 训练到下游策略训练的硬件流程指南，并有部署参考和数据格式说明。代码使用 Apache-2.0。README 明确预训练策略与硬件数据不随仓库发布，因此“代码可用”不等于数据和 checkpoint 可直接复现实验。
