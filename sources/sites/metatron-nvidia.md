# METATRON：可提示式人形全身执行层

- **类型：** NVIDIA 官方项目页
- **项目页：** https://research.nvidia.com/labs/par/projects/metatron/
- **论文：** https://arxiv.org/abs/2610.38027 （v1，2026-10-27）
- **框架代码：** https://github.com/NVlabs/ProtoMotions
- **模型：** https://huggingface.co/nvidia/Metatron-v1
- **官方运行文档：** https://nvlabs.github.io/ProtoMotions/getting_started/pretrained_models.html
- **核查日期：** 2026-10-31。

## 已核查的项目页信息

先以完整参考动作训练跟踪器，再通过掩码目标的 on-policy 蒸馏得到策略；部署时冻结权重，技能代码提供目标，策略输出关节目标。提示是结构化运动目标，不是自然语言控制指令。实机 Unitree G1 与 MuJoCo 仿真演示需分别看待；跨栏、保龄球及网球为仿真例子。

官方文档列出公开的 Metatron v1 G1 模型和 ProtoMotions 推理入口；具体代码与权重使用范围以各自许可证及模型卡为准。

## 阻塞与待完成

本次无法获取论文全文：PDF 超出检索工具大小限制，HTML 入口不可用；尚未通读论文方法、实验、局限及附录。源码运行链路也尚未逐文件核验，因此不能绘制声称基于全文的流程图或源码时序图。需完成两项核查、建立实体与来源交叉链接并执行仓库预检后再将 PR 标记为可审核。
