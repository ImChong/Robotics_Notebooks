# FloMo 官方项目页

> 来源归档（site）；2026-11-19 再次核对原始项目页面及资源入口

- **标题：** FloMo: 3D Scene Flow as a World Action Model Intermediate for Learning from Human Video
- **类型：** site
- **链接：** <https://flomo-wam.github.io/>
- **论文：** <https://arxiv.org/html/2611.05062>
- **代码：** <https://github.com/pairlab/flomo>
- **模型权重：** <https://huggingface.co/shitijgovil/FloMo>
- **入库日期：** 2026-11-05；项目页核对：2026-11-19
- **核对范围：** 项目页标题与作者、Methods、In-/Out-of-distribution、RoboTwin、Human-video ablation、Code/Paper 资源。
- **通读：** 已完成
- **沉淀到 wiki：** [FloMo](../../wiki/entities/paper-flomo.md)

## 开源状态

**代码与模型权重已公开。** 项目页直链官方 [pairlab/flomo](https://github.com/pairlab/flomo)，该仓库 README 提供训练、预处理、推理脚本与 MA/VA/VMA 权重下载链接；不等于真实 YAM 硬件/内部数据均公开。

## 项目页可复核的指标

- 真机已见任务（Oven / Toolbox / Corn）：FloMo 91%、VMA 51%、VA 40%，各任务 15 次。
- 3 类真机 OOD：FloMo 18/30（60%）、DreamZero 和 AMPLIFY 各 6/30（20%）、VA/VMA 各 2/30（约 7%）。
- RoboTwin Easy 5 任务均值：FloMo 63.6%、π₀.₅ 44.8%、FAST-WAM 41.0%、BC 19.2%。
- 移除人类视频：OOD 成功率从 18/30 降至 1/15，测试次数不同。
