# SADL 官方项目主页归档

- **类型：** site
- **项目：** https://zju3dv.github.io/SADL/
- **论文：** https://arxiv.org/abs/2611.21596
- **代码：** https://github.com/zju3dv/SADL
- **模型与数据：** https://huggingface.co/DavidSunok/SADL
- **核对日期：** 2026-11-20
- **开放状态：** 已开源；项目页提供论文、GitHub、视频入口，公开训练/评测脚本和模型/数据获取链接。
- **核对范围：** 核对项目页的全部主要板块（作者机构、方法可视化、主结果、坍塌修复、图像任务、引用和代码入口）；不将互动演示值当作独立复现实测。
- **通读：** 已完成

## 项目页确认内容

- Encoder H 的公共均值可能被 BN projector 过滤，单凭 Z 的健康度会遗漏上游坍塌。
- SADL 从 Z 的 Demand Regularizer 和分布中抽取需求 q，测量 H 激励与 H→Z 传输的供给 k，只在不足时增加 Encoder 约束。
- 官网报告坍塌诊断 50%→0%，相对基线的 H 有效秩 3.4–6.4 倍，部分 INTACT 无搜索控制设置超过 95%。
- 代码、checkpoint 和数据有官方公开入口；本次未本地运行。

## 关联

- [论文归档](../papers/sadl-arxiv-2611-21596.md)
- [代码归档](../repos/sadl-zju3dv.md)
- [唯一论文节点](../../wiki/entities/paper-sadl-2611-21596.md)
