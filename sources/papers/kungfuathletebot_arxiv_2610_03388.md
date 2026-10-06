# KungfuAthleteBot: Learning High-Dynamic Humanoid Motion from Video with Unified Robust Recovery

> 来源归档（paper；核对 arXiv HTML v1、项目页、GitHub README 与 Hugging Face 数据卡；2026-10-06）

- **作者：** Zhongxiang Lei, Lulu Cao, Xuyang Wang, Tianyi Qian, Jinyan Liu, Xuesong Li
- **机构：** Beijing Institute of Technology；QIYUAN Lab
- **论文：** <https://arxiv.org/abs/2610.03388> · [HTML v1](https://arxiv.org/html/2610.03388v1)
- **项目页：** <https://kungfuathletebot.github.io/>
- **代码：** <https://github.com/NPCLEI/KungFuAthleteBot>
- **数据：** <https://huggingface.co/datasets/LuluCao/KungfuAthleteBot>
- **前序预印本：** *A Kung Fu Athlete Bot That Can Do It All Day*, arXiv:2602.13656；本次按 2610.03388 更新既有项目实体，不重复建页。
- **一句话说明：** 从武术公开视频重建运动参考，通过物理引导轨迹修复、伪低动能采样和三阶段课程补偿视频缺乏驱动力信息的问题，再让同一 G1 策略联合跟踪、抗扰与跌倒恢复。

## 核心摘录

1. **三类数据失效模式。** 单目视频重建会出现腾空根高度漂移、落地穿透与高频抖动；视频不包含执行力/力矩，直接跟踪可在动力学上不可行；常规跟踪策略又不建模失败后的恢复。
2. **轨迹修复与训练。** 对腾空与着地片段做物理引导抛物线根轨迹修正，减少悬浮、穿地和抖动；用 physics-driven pseudo-low-kinetic-energy（LKE）采样，把训练初始化偏向动力学可行状态，并配合三阶段课程，避免误差驱动反复从不可能的空中姿态重启。
3. **统一目标。** disturbance rejection、motion tracking 和 fall recovery 由同一策略训练，不需要恢复示范或手动模式切换；论文在 Unitree G1 真机报告任意跌倒约 0.7 秒恢复并返回跟踪。
4. **数据。** 新论文附录 C 与当前 Hugging Face 卡给出 992 条：Ground 822、Jump 170；从 197 个原始公开视频切出 1,726 段。原始运动员视频不再分发。
5. **可复现边界。** 官方 GitHub 含可运行的 retarget / height-adjustment 与 Unitree RL Mjlab 训练/回放代码，并列有训练配置、1307 跌倒恢复 checkpoint 与真实部署说明；新论文称项目页包含 848 条 30 fps robot qpos 数据和训练文件。HF 当前卡为 992 条，许可元数据标为 Apache-2.0；论文则称代码、数据与 checkpoint 将在接收后按 MIT 发布。仓库和项目页仍有旧版 848 样本表述，因此需按具体文件/版本核对，不要将论文计划许可视作所有现存资产的统一授权。

## 对 Wiki 的映射

- [KungfuAthleteBot 论文实体](../../wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md)
- [代码仓库来源归档](../repos/kungfuathletebot.md)
- [项目页来源归档](../sites/kungfuathletebot.md)
- [Hugging Face 数据集归档](../datasets/kungfuathletebot-hf.md)
- [平衡与恢复](../../wiki/tasks/balance-recovery.md)；[动作重定向](../../wiki/concepts/motion-retargeting.md)

## 参考来源（原始）

- arXiv HTML v1：<https://arxiv.org/html/2610.03388v1>
- 官方项目页：<https://kungfuathletebot.github.io/>
- 官方代码仓库：<https://github.com/NPCLEI/KungFuAthleteBot>
- 官方 Hugging Face 数据卡：<https://huggingface.co/datasets/LuluCao/KungfuAthleteBot>
- 前序版本 arXiv:2602.13656：<https://arxiv.org/abs/2602.13656>
