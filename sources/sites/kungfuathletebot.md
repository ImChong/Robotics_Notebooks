# KungfuAthleteBot 项目页

> 来源归档（site；核对日期：2026-10-06）

- **论文（新版）：** <https://arxiv.org/abs/2610.03388>
- **项目页：** <https://kungfuathletebot.github.io/>
- **代码：** <https://github.com/NPCLEI/KungFuAthleteBot>（可运行的 retarget / height-adjustment 与 Unitree RL Mjlab 训练入口）
- **数据：** <https://huggingface.co/datasets/LuluCao/KungfuAthleteBot>
- **源码状态：** 已开源部分训练、动作处理与部署流程；README 勾选数据、height-adjustment、training、FastSAC、1307 恢复 checkpoint 与 real deployment。项目页仍保留旧版 arXiv 2602.13656 / 848 样本简介，且写有更多模型将发布。
- **数据状态：** HF 当前数据卡计 992 样本（Ground 822、Jump 170），标记 Apache-2.0；原始武术视频因人物隐私/素材许可不再分发。新论文附录 E 对应写 848 筛选样本、30 fps robot qpos，并称完整资料将在接收后 MIT 发布。公开页面之间版本和许可口径不一致，使用者应逐资产核对。

## 项目页可确认的信息

- 论文和项目沿用 GVHMR 人体重建与 GMR 重定向；官方仓库公开根高度调整脚本及演示数据结构。
- 数据卡上游材料是 197 段公开视频，自动分段为 1,726 子片段；卡片目前按 Ground/Jump 汇总为 992 个 robot motion 样本。
- Ground 子集较稳定；Jump 子集仍有视频源噪声，仓库明确提醒真机直接使用高动态 Jump 数据存在硬件风险。
- 新论文报告 G1 真机恢复约 0.7 秒；此数值是该策略评测结果，不应外推为通用跌倒安全保证。

## 互链

- [论文来源](../papers/kungfuathletebot_arxiv_2610_03388.md)
- [代码仓库来源](../repos/kungfuathletebot.md)
- [Hugging Face 数据归档](../datasets/kungfuathletebot-hf.md)
- [Wiki 论文实体](../../wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md)
