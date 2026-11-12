# RoboTTT: Context Scaling for Robot Policies（arXiv:2607.15275）

> 来源归档（paper · arXiv 全文通读）

- **标题：** RoboTTT: Context Scaling for Robot Policies
- **类型：** paper
- **作者：** Yunfan Jiang, Yevgen Chebotar, Ruijie Zheng, Fengyuan Hu, Yunhao Ge, Jimmy Wu, Tianyuan Dai, Scott Reed, Li Fei-Fei, Yuke Zhu, Linxi "Jim" Fan（NVIDIA / Stanford / UT Austin）
- **arXiv：** <https://arxiv.org/abs/2607.15275>（v1，2026-07-16）
- **PDF：** <https://arxiv.org/pdf/2607.15275v1>
- **项目页：** <https://research.nvidia.com/labs/gear/robottt/>
- **代码：** 正文未给出代码仓库链接（致谢只提到「help with the release」），本归档不写 code 地址
- **入库日期：** 2026-11-12
- **核对范围：** 读了 v1 PDF 正文第 1-6 节全部文字、图 1-12 的图注、表 1-3、附录 A（模型 / 训练 / 部署）与附录 B（任务定义与评分细则）；图 7 柱状图的逐任务数值是从图上目测读取，仅作约数；参考文献列表只扫读未逐条核对；未读项目页视频，未运行任何代码。
- **通读：** 已完成

## 论文自述的核心内容（章节 / 图表出处）

1. **做什么（摘要，§1）：** 在机器人基础模型（VLA）里集成 Test-Time Training（TTT）层，循环状态是 fast weights（训练与推理时都由梯度下降更新），把 visuomotor 上下文扩到 8K timesteps，作者称比现有策略高三个数量级且推理延迟不随上下文增长（§1 摘要措辞；正文未见延迟实测曲线）。
2. **TTT 机制（§2，式 1-2）：** `W_t ← W_{t-1} − η∇ L_FW(f_{W_{t-1}}(K_t), V_t)`，`O_t = f_{W_t}(Q_t)`；`L_FW` 为均方误差；θ_Q、θ_K、θ_V 与 W_0 由外层任务损失学习。
3. **架构（§3.1，图 2、3，式 3）：** 在 GR00T N1.7 的 DiT 动作头每层 self/cross-attention 之后加 TTT 层（attention 处理单步，TTT 处理跨时间）；VL token 不直接过 TTT，而用 N=16 个 register token 携带 VL 信息跨时间；`O = tanh(α) ⊙ O_TTT + O_attn`，α 初始化 0.001。
4. **训练（§3.2，式 4-5，图 4）：** sequence action forcing（每个 action chunk 独立采样流匹配噪声水平，τ_t = s(1−u)，u~Beta(1.5,1)，s=0.999），论文称不用则训练不稳定（「possibly because」属推测）；TBPTT 分段反传，fast weights 跨段传递、梯度在段边界截断，显存由段长决定。
5. **从上下文学习（§3.3，图 6）：** 对选定 timestep 掩掉流匹配损失，仅更新 fast weights。(a) 人视频 + 机器人轨迹配对做 one-shot 模仿；(b) DAgger Distillation：全部执行动作（含次优机器人动作）更新 fast weights，损失只算在人工纠正上，作者称其为 Algorithm Distillation 的机器人实例。
6. **实现（§3.4、附录 A）：** 16 个 DiT 层各加一个 TTT 层，fast model 为两层 MLP（GeLU）；DiT 原 538M 参数，每个 TTT 层约 +10M，总计 690M；内层学习率在常数基值 0.1 上学习；RoPE θ=10000；预训练混合桌面双臂机器人数据与第一视角人类视频（Ego 数据引文 81），上下文逐步增至目标长度，30K 步，16 张 GB200；预训练只调新增序列层（TTT 或 GDN），冻结 GR00T N1.7 其余部分；下游后训练统一 1K 上下文、20K 步、全参数微调；部署在 YAM 双臂桌面机器人，4 个 RealSense D405（顶、底、左腕、右腕，480p），RTX 5090 工作站，30Hz。
7. **评测设置（§4，附录 B）：** 三个装配任务，YAM 双臂：Pup Go Car（平均 2 分钟，8 小时数据）、Circuit（平均 1 分钟，6 小时数据，约 80 种配置，20 种训练、60 种测试）、Gear Bot（平均 5 分钟，十阶段，5 小时数据）。基线：GR00T N1.7（单步）、GR00T N1.7 Hist.（+1 历史帧）、GDN（把 TTT 层换成 Gated DeltaNet，层位置、门控、参数量匹配）。每策略评测 20 次（Gear Bot 与 Circuit 的人视频设定为 10 次），初始物体摆放在方法间复现一致；指标为完全成功次数与按任务评分表归一化到 [0,1] 的完成分。
8. **主结果（§4，图 7、表 1）：** 平均完成分 RoboTTT 79%、GR00T N1.7 42%（+87%）、最佳基线 GDN 56%（+41%）。完全成功次数（表 1）：Pup Go Car 9/20 vs 3/20(N1.7) / 0/20(Hist.) / 3/20(GDN)；Circuit 13/20 vs 3/20 / 8/20 / 8/20；Gear Bot 2/10 vs 三个基线均 0/10。图 7 目测各任务完成分约：Pup Go Car 0.89 / 0.57 / 0.395 / 0.57；Circuit 0.78 / 0.23 / 0.55 / 0.53；Gear Bot 0.71 / 0.47 / 0.54 / 0.59（顺序 RoboTTT / N1.7 / Hist. / GDN，图 7 标 Circuit 为 Unseen Configurations）。文中明确写出的单任务数字只有 Pup Go Car 上 Hist. 39.5% vs N1.7 57%。
9. **上下文 scaling（§4，图 8）：** 预训练上下文 128 → 8K，RoboTTT 平均完成分稳步上升，8K 时 71.5%，比 1K（43.9%）高 63%，比最佳短上下文基线 Hist.（45.6%）高 57%，作者称「无饱和迹象」；GDN 无此趋势；低于 1K 时 RoboTTT 仍优于同长度 GDN 但不及更长变体，作者归因于 rollout 时长超过训练上下文（1K ≈ 半分钟）。图注说明图 8 的评测早于 Pup Go Car 主结果中使用的 DAgger 训练。摘要写 62%、§1 写 63%，两者均由 71.5 / 43.9 取整而来。
10. **one-shot 人视频模仿（§4，图 9，表 2）：** Circuit，同一提示词 "assemble circuit"，目标配置只能由上下文视频确定；训练时每个训练配置收集 5-20 条人类视频（机器人静止、人手装配）；10 次试验 RoboTTT 6/10（完成分 65%），GDN 0/10（33%）。GR00T N1.7 及 Hist. 因无法容纳人视频未对比。
11. **扰动鲁棒性（§4，表 3）：** Pup Go Car，人为取走已装的黄色车顶或轮胎，每条件 20 次，成功重装次数 roof：RoboTTT 15、N1.7 10、Hist. 3、GDN 13；tire：18、11、5、18。扰动数据 30 分钟与任务数据 co-train。§1 的「83% vs 53%」与表 3 合计 33/40 vs 21/40（N1.7）一致，作者在 §1 称之为最佳短上下文基线，Hist. 实为 8/40，故 53% 对应的是 N1.7（单步）。
12. **DAgger Distillation（§4，图 10、11）：** Pup Go Car，100 条 DAgger 轨迹（RoboTTT 与 GR00T N1.7 各采 50 条）；标准 DAgger（只用人工纠正）平均 +9%（序列模型 +13%）；DAgger Distillation 平均 +33%（RoboTTT +36%，GDN +29%）；GR00T N1.7 用整条轨迹（含机器人动作）与只用纠正都是 57%，作者据此说次优动作的价值在于作上下文。
13. **消融（§4，图 12）：** 去掉 sequence action forcing 显著变差（机器人无法有意义推进）；TTT-Linear 比 MLP fast model 差 27%（但仍好于 GR00T N1.7）；只过状态 token 的 TTT 基础上加动作 token 相对 +23%、再加 register token 相对 +18%；给 GR00T N1.7 加同数量 register token 无帮助。图 12 数值未在正文列出，仅取正文百分比。
14. **作者承认的局限（§6）：** 训练上下文变长使训练成本上升（可考虑 TNT 等更新的 TTT 训练技术）；可探索面向机器人的 TTT 目标；并非所有部署失败模式都能处理，可与强化学习结合。附录 A 另注明更复杂的测试时优化器（如 Muon）留待未来。

## 与 wiki 现有页对不上之处（供核对）

- 旧页把任务时长写成「Pup Go Car ~5 min」；论文是 Pup Go Car 2 分钟、Circuit 1 分钟、Gear Bot 5 分钟（§4、图 5）。
- 旧页逐任务完成分 Gear Bot「~78 vs ~35」、Circuit「~71 vs ~23」与图 7 目测不符（Circuit 约 0.78 vs 0.23，Gear Bot 约 0.71 vs 0.47），且 Circuit 本来是一分钟任务。
- 旧页「长程双臂装配 +87%」在论文里是三任务平均完成分相对单步基线（79% vs 42%），不是单个长程任务上的提升。

## 对 wiki 的映射

- [paper-robottt-test-time-training-vla-context](../../wiki/entities/paper-robottt-test-time-training-vla-context.md)
- 早期摘录归档（项目页 + 公众号策展，未通读论文）：[robottt_nvidia_gear.md](robottt_nvidia_gear.md)
