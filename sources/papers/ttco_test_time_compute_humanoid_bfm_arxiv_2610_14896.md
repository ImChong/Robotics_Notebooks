# TTCO（arXiv:2610.14896）

> 来源归档（paper；2026-10-16 核验更新）

- **标题：** Test-Time Compute for Humanoid Behavior Foundation Models
- **类型：** paper
- **作者：** Weiji Xie、Junsong Wu、Zibo Zhou、Yueru Chen、Xingyi Wang、Zhenhua Xiong、Weinan Zhang、Chenjia Bai
- **机构：** 中国电信人工智能研究院（TeleAI）、上海交通大学、浙江大学、Gamma Robotics
- **提交日期：** 2026-10-10；v2 更新：2026-10-14
- **arXiv：** <https://arxiv.org/abs/2610.14896>
- **PDF：** <https://arxiv.org/pdf/2610.14896>
- **项目页：** <https://bfmttc.github.io/>
- **官方代码：** <https://github.com/Gamma-Robot/bfm_ttc>
- **入库日期：** 2026-10-14；源码核验更新：2026-10-16
- **线索来源：** [具身智能小站 41 篇论文速览](../raw/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)
- **一句话说明：** 在闭环仿真中为冻结的人形行为基础模型搜索输入命令，并利用机器人轨迹生成 BFM 专属动作标签。

## 开源状态

- **已开源（2026-10-16 核验）：** 项目页直接链接 [Gamma-Robot/bfm_ttc](https://github.com/Gamma-Robot/bfm_ttc)，仓库公开提供 TTCO、采样优化器、BFM adapters、仿真运行入口及 surrogate cost model 实验代码。
- **复现条件：** README 要求另行准备 G1 robot assets 与 HEFT-PMG checkpoint；目前未发现项目专属预训练权重或独立数据集下载页。仓库主代码 Apache-2.0，SPIDER sampler 保留 CC BY-NC 4.0 上游许可。
- 归档详见 [项目页核验](../sites/ttco-test-time-compute-humanoid-bfm.md) 与 [源码仓库核验](../repos/bfm_ttc.md)。

## 核心摘录（论文与项目页）

1. **测试时命令优化（TTCO）：** 对目标 motion 采样候选输入，在冻结 BFM 与机器人动力学构成的闭环仿真中评估；选取低代价候选的短段执行，然后重规划。
2. **三种 BFM 的误差改善：** 在 AMASS 的 12,273 条 motion 上评估 HEFT、SONIC、TWIST2。相对将参考 motion 直接作为命令，仿真平均全局身体位置误差分别由 27.1→5.7 cm、61.5→12.7 cm、109.5→17.3 cm，下降 79–84%。
3. **预算配置：** 论文分析测试时计算扩展，并拟合 surrogate cost model 近似计算最优分配；项目页展示平均 cost regret 4.02%，预算 K=256 增至 K=8192 时 cost 降低 58.0%。
4. **示范重标注：** TTCO 可从记录的机器人轨迹离线生成 BFM 专属命令标签，不需要原 teleoperation body-motion commands；用于训练高层 loco-manipulation policy 与跨 BFM 复用示范。
5. **真机策略评测：** 项目页报告 SONIC 示范上微调 GR00T N1.7，四任务平均成功率：TTCO body-action labels 64.2%、原始命令标签 60.4%、robot-state labels 2.5%（每任务 / 每标签来源 60 rollouts）。逐任务结果有涨有跌，不能把均值表述成逐项稳定提升。

## 对 wiki 的映射

- [TTCO 统一论文实体页](../../wiki/entities/paper-ttco-test-time-compute-humanoid-bfm.md)
- [Behavior Foundation Model 概念页](../../wiki/concepts/behavior-foundation-model.md)
- [Loco-Manipulation 任务页](../../wiki/tasks/loco-manipulation.md)

## 参考来源（原始）

- arXiv 摘要及版本记录：<https://arxiv.org/abs/2610.14896>
- [官方项目页](../sites/ttco-test-time-compute-humanoid-bfm.md)
- [官方代码仓库](../repos/bfm_ttc.md)
- 公众号线索：[原始周报](../raw/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)
