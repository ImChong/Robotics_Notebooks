# TouchScale: 500 Hours of Human Vision and Touch for Visual-Tactile Learning

> 来源归档（ingest · arXiv 预印本）

- **类型：** paper / dataset
- **arXiv：** <https://arxiv.org/abs/2610.10288>
- **版本/日期：** v1 于 2026-10-07 提交；v2 于 2026-10-08 修订
- **作者：** Dayou Li、Hao Wang、Qianqian Yang、Zihao Zhu、Haoquan Fang、Ziyao Zeng、Yan Han、Zihan Wang、Yan Wang、Baoru Huang、Dilin Wang、Kenji Shimada、Yiyue Luo、Manling Li、Teresa Lv、Mustafa Mukadam、Rakesh Ranjan、Ruohan Zhang、Qi He、Changliu Liu、Xu Chen、Marco Pavone、Bangya Liu、Jiachen Li、Masayoshi Tomizuka、Zhiwen Fan
- **项目页：** <https://touch-scale.github.io/>
- **数据集：** <https://huggingface.co/datasets/2077AIDataFoundation/TouchScale>
- **作者机构：** Texas A&M University、Google DeepMind、CMU、Stanford University、Yale University、Microsoft、Overfit Lab、NVIDIA、University of Liverpool、Meta、University of Washington、Northwestern University、Sony、Georgia Tech、UC Berkeley（论文 v2 所列）
- **核查日期：** 2026-10-10
- **沉淀到 wiki：** [TouchScale](../../wiki/entities/touchscale.md)

## 论文贡献

TouchScale 是一个统一可穿戴采集流程下的人类视觉-触觉交互数据集。每条记录对齐头戴 RGB-D、左右手腕 RGB 视频和双手全手触觉测量；论文描述全量约 500 小时、约 87,000 episodes、约 2,000 个任务描述和 1,500+ 物体。每只手的触觉手套包含 880 个 taxels，覆盖手指和手掌。

作者在跨传感器触觉预测、动作识别和真实机器人接触丰富操作上进行评估。涉及机器人策略时，TouchScale 用作视觉-触觉 mid-training 数据，不对人手动作做机器人动作重定向，再用机器人示范进行后训练。

## 指标口径说明

摘要将 full-data 规模实验的跨传感器 contact IoU 从 0.134 提升至 0.383；论文另有一个 size-matched 约 16 小时对照，TouchScale 为 0.181、EgoTouch 为 0.134。两者回答不同问题，不应混为同一比较。四个接触丰富真机任务平均成功率从 22.5% 到 57.5% 是论文所报告的 mid-training + robot post-training 设置，不能据此推断任意任务或硬件均有相同提升。

## 一手入口

- [arXiv v2 摘要与版本记录](https://arxiv.org/abs/2610.10288)
- [arXiv HTML v2](https://arxiv.org/html/2610.10288v2)
- [论文 PDF](https://arxiv.org/pdf/2610.10288)
- [TouchScale 数据集归档](../datasets/touchscale_huggingface.md)
- [官方项目主页](https://touch-scale.github.io/)
