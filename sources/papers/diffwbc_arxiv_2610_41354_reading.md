# DiffWBC 原文通读（arXiv:2610.41354v1）

> 来源归档（paper；支撑 `wiki/entities/paper-diffwbc.md`）

- **标题：** DiffWBC: Unlocking Steerable General Humanoid Motion Tracking via One-Step Diffusion Whole-Body Control
- **作者：** Yifei Huang, Yazhao Yang, Weijia Li, Zhiheng Ma, Dian Shen
- **机构：** 东南大学（Southeast University）；南方科技大学（Southern University of Science and Technology）；香港理工大学（The Hong Kong Polytechnic University）；深圳理工大学（Shenzhen University of Advanced Technology）
- **类型：** paper / humanoid / whole-body-control / motion-tracking / generative-policy
- **发表：** arXiv:2610.41354v1，2026-10-30（预印本；非已确认的会议论文）
- **原文：** https://arxiv.org/abs/2610.41354 （[HTML 全文](https://arxiv.org/html/2610.41354) / [PDF](https://arxiv.org/pdf/2610.41354)）
- **项目页：** https://diffwbc.github.io/
- **入库日期：** 2026-11-10
- **一句话说明：** 在 50 Hz 低层全身控制器内部用单步生成器预测未来状态与动作，支持冻结权重后的代价梯度引导、分组参考遮蔽与无仿真失效挖掘。
- **核对范围：** 已通读 arXiv 2610.41354v1 正文摘要、§1–§5（相关工作、完整方法、四组实验与局限）、Tables 1–6、Figures 1–3 图注、Appendix A–C（Table 7–12、steering 消融、VR 三点控制、失败评分推导与成本），核对项目页；参考文献核对涉及的对比对象，未逐篇阅读全文。
- **通读：** 已完成

## 机制摘录（以论文 v1 为准）

1. **输入与身体分组（§3.1）：** 10 帧未来参考（每帧间隔 0.1 s），每帧 38 维：root 9 / legs 12 / upper body 17，包含 29 个驱动关节角。各组 MLP 时序编码后求和，再经 FSQ 编为 2 个 32 维 token。训练时各组独立以 0.3 概率遮蔽，整条参考以 0.1 概率遮蔽。
2. **一步生成（§3.2）：** Transformer 4 层，hidden 512，8 heads，22.8M 参数；本体 token + 2 个参考 token + 24 个轨迹 token。每个轨迹 token 携带 73 维物理状态（根状态与 29 关节 q / dq）及 29 维目标关节角。前 16 个 50 Hz 时间步稠密预测，后 8 个每 4 tick 取样，预测总视野 0.96 s。训练从加噪轨迹直接回归清洁未来；常规执行时从初始高斯噪声只需一次前向。
3. **执行（§3.2）：** 逆动力学 action head 读取本体状态与前 4 个预测状态，给出关节控制命令；下一控制 tick 使用实测本体状态重新生成计划，而不是把整段计划开环执行。
4. **双训练路径（§3.3–3.4）：** 在线 DAgger 蒸馏 MimicLite Huge teacher：4096 个并行仿真环境，78,842 个 BONES-SEED / AMASS / LAFAN 运动序列、约 170 h；4 张 RTX 4090 运行 8 h = 32 GPU-hours。另可不经 teacher，用 learned neural world model 的 32 步 imagined rollouts 做 model-based RL，60,000 iter 约 20 GPU-hours。
5. **测试时引导（§3.5–3.7）：** 用几何避障 SDF、空间末端目标、力矩限制、托盘倾角或速度误差等可微 cost，反传到输入噪声而不更新模型参数。默认两次生成与两步梯度下降；最多 12 步用于高难末端到达。body-group mask 将某些肢体从跟踪约束释放给 steering，其他肢体维持原参考。可选世界模型投影修复关节角调整引起的动力学不一致。

## 实验事实（§4 与附录）

- **跟踪（Table 2）：** 统一 MuJoCo sim2sim；两套各 150 条 held-out 窗口。RL from scratch DiffWBC 在 AMASS 的 local error 26.5 mm，BONES 20.1 mm，BONES success 97.3%；蒸馏版 BONES 24.7 mm。与 SONIC、HEFT、HoloMotion、Humanoid-GPT、MimicLite Huge 对照，指标只适用于该统一 harness 与窗口设置。
- **真机（§4.2）：** Unitree G1（29 actuated joints），9 条全身动作各试 3 次；论文报告这 27 次均完成且未摔倒，有松弛安全绳。真机实验是普通 motion tracking，未报告 steering 真机演示。
- **时延（§4.2、Table 3）：** 单张 RTX 4090 **普通推理** 3.8 ms；默认 steering 20.3 ms，难目标可至 55.1 ms，需注意 50 Hz 的 20 ms 控制周期。不能写成「全部训练与引导均为单卡实时」。
- **障碍与到达（Table 3）：** 1.25 m ceiling 下 steered clean pass 0.84，对比闭环编辑参考 0.10；障碍 bars 可行率 0.68 vs 0.57；高难 reaching hit 0.75 vs 0.60。均为仿真。
- **多目标（Table 4）：** 左膝限矩 15% 时 survival 0.71→0.79；托盘滚珠丢失 0.17→0.08；速度误差 0.159→0.112 m/s。均为仿真。
- **mask（Table 5）：** 遮蔽 upper body 后 reaching hit 0.54→0.67；仅给上半身参考仍有 0.926 survival，但局部误差升至 48.2 mm。
- **失效挖掘（§4.5、Appendix C）：** 7,884 clip 扫描含 537 失败；对 8 份噪声计划做 spread，top 2% 精度 0.91（AUC 0.83）。在筛选出的 329 条 clip 上微调，修复 baseline failures 的 50%，1.9% 已掌握 clip 退化；实验为单种子/指定数据协议，非形式化失效保证。
- **附录重要取舍：** scratch 跟踪局部误差较好，但 test-time steering 能力弱于蒸馏版（Table 7：1.25m ceiling success 0.00 vs 0.84）；对于高接触力、不连续接触仍需显式 wrench 模型；轨迹 steering 的动力学一致性并非自动保证。

## 一手资料版本差异与开源状态

- **论文身份优先：** arXiv v1 题名 DiffWBC；同域项目页当前仍显示匿名旧名 OneSteer，部分 BONES 误差表中数字为 **22.6 mm**，与论文 v1 的 **20.1 mm** 不同。笔记的论文指标均以 v1 为准，不合并二者不同版本数值。
- **开源：** 截至 2026-11-10 项目页仍显示 “Code (after review)” 且未给出公开可运行源码仓库/权重。无法基于源码核对真实入口，不能绘制源码运行时序图。
- **验收边界：** 论文的测试时 steering、故障补偿与失效识别为仿真研究；真机验证只覆盖基础跟踪。

## 对 wiki 的映射

- [DiffWBC 论文实体页](../../wiki/entities/paper-diffwbc.md)
- [SONIC 通用跟踪](../../wiki/methods/sonic-motion-tracking.md)
- [YAHMP 跟踪消融](../../wiki/entities/paper-yahmp.md)

## 参考来源

- https://arxiv.org/abs/2610.41354
- https://arxiv.org/html/2610.41354
- https://diffwbc.github.io/
