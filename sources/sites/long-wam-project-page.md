# Long-WAM 官方项目页

> 来源归档（site；核对 NVIDIA 官方 Long-WAM 页面；2026-10-08）

- **官网：** <https://nvlabs.github.io/LongLive/Long-WAM/>
- **论文：** <https://arxiv.org/abs/2610.10528>
- **源码：** <https://github.com/NVlabs/LongLive/tree/main/Long-WAM>
- **权重：** <https://huggingface.co/collections/Efficient-Large-Model/long-wam>
- **标题：** *Long-WAM: Scaling the Context of World-Action Models*
- **团队：** Wei Huang、Bohan Zhang 共同第一作者；作者单位含 NVIDIA、MIT、The University of Hong Kong、UC San Diego。

## 页面内容

- **问题：** WAM 的可用规划历史应随任务阶段变化；Long-WAM 扩展观测上下文而不同比例增加未来预测或动作块。
- **数据：** 页面称 LongLive2.0-Robot 基于 LongLive-2.0 checkpoint 继续训练，汇集约 10,000 个窗口等价小时机器人/第一视角视频，包含 RoVid-X、AgiBot World、EgoDex、EgoVerse、VITRA；视频预测适配无需动作标签。
- **架构：** 因果视频专家预测未来视觉 latent；动作专家据历史与预测未来生成动作。视频专家不读取动作 token，推理无需解码完整视频。
- **结果：** RoboCasa GR-1 历史从 0 秒到 19.2 秒时 SR 报告为 63.3%→78.7%；页面另列 2.4 秒点位 66.3%。动态杯叠放真机报告 19/20。
- **延迟：** 页面报告 RTX 5090 上 107.4 ms/action chunk，包含未来 latent 预测。
- **演示边界：** 标作 future prediction 的生成视频是模型预测，不代表机器人执行该预测并完成任务。

## 引用边界

本页的指标、数据规模与硬件数据均为作者/项目团队报告，本归档保留原始来源而非独立复现实验。须结合论文及 Long-WAM README/复现文档阅读 benchmark 结果。

## 沉淀到 Wiki

- [Long-WAM 独立详情节点](../../wiki/entities/paper-long-wam-scaling-context.md)
- [论文来源归档](../papers/long-wam-arxiv-2610-10528.md)
- [官方代码归档](../repos/nvlabs-longlive-long-wam.md)
