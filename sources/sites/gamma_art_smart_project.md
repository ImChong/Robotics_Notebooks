# SMART / Gamma-Art 官方项目主页来源

- **项目：** SMART: Zero-Shot Sim-to-Real Articulated Object Manipulation via Large-Scale Synthetic Pretraining
- **论文：** https://arxiv.org/abs/2610.07652
- **项目主页：** https://teamillusion-smart.github.io/
- **数据：** https://huggingface.co/datasets/TeleEmbodied/SMART-Data
- **模型：** https://huggingface.co/TeleEmbodied/SMART-VLA
- **共享模型代码：** https://github.com/TeleHuman/PRTS
- **入库日期：** 2026-11-11
- **核验：** 项目 URL 由论文和官方 HF 页面交叉指向；本轮网页读取项目站点受限，不能据此推断其全部资源链接，数据/模型/仓库开放状态逐项以官方 HF 和 GitHub 为准。

## 项目定位

围绕铰接物体操作，用 SMART-Sim 的结构约束运动生成、任务合成、分布式渲染与域随机化收集大规模仿真视觉动作数据，再训练 SMART-VLA 在真机零样本迁移。

## 开放状态

- **已发布：** SMART-Data 数据（LeRobot v3.0）和 SMART-VLA 权重。
- **已有可运行的共享实现：** TeleHuman/PRTS 提供通用模型后训练、模型加载及 WebSocket 策略服务；不是 SMART-Sim 专属开源仓库。
- **仍未核实可获得：** SMART-Sim 完整物体关节标注、agent 任务编排、分布式规划与渲染生成源码。论文没有提供这些模块的可复现入口。

## 关联

- [论文原文归档](../papers/gamma_art_smart_arxiv_2610_07652.md)
- [数据集归档](../datasets/gamma_art_smart_data.md)
- [共享代码入口归档](../repos/telehuman_prts_smart_vla.md)
- [知识实体](../../wiki/entities/paper-gamma-art-smart.md)
