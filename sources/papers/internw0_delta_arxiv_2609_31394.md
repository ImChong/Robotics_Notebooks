# InternW0-Δ — 论文一手资料归档

- **标题：** InternW0-Δ: A World Action Model Bridging Predictive Dynamics and Actions with 20K+ Hours of Open Data
- **作者：** Xingyu Miao、Zizun Li、Baole Fang、Kaiwen Song、Tenghui Wang 等（完整作者列表见 arXiv）
- **机构：** Shanghai AI Laboratory（论文团队署名 Physical Intelligence Team）
- **提交日期：** 2026-09-25，arXiv:2609.31394v1
- **类别 / 许可：** cs.RO、cs.CV；论文标注 CC BY-NC-SA 4.0
- **论文：** <https://arxiv.org/abs/2609.31394>
- **HTML：** <https://arxiv.org/html/2609.31394v1>
- **官方项目：** <https://internrobotics.github.io/InternW0-Delta/>
- **策展节点：** [InternW0-Δ](../../wiki/entities/paper-internw0-delta.md)

## 论文贡献摘要

1. **定向 World–Action MoT：** 预训练视频专家与动作专家独立参数化，通过 mask 约束的联合 self-attention 交互；冻结 VLM 为动作专家提供场景和指令语义。
2. **Causal Imprint：** learnable Δ tokens 从可观测上下文学习未来相关变化。训练时用未来潜变量差分直接监督，并以 stop-gradient 对齐未来视频特征；动作路径不能读取未来帧，因此推理不必生成未来视频。
3. **训练期 4D-aware distillation：** 冻结 Track4World 教师输出离线缓存的 1430D 几何/运动描述子；query decoder 将这些先验蒸馏给视频专家，教师和蒸馏头不进入策略推理。
4. **跨源数据管线：** 机器人示范、UMI、人类第一视角和 Ego2Robot 合并为带 mask 的 80D state/action 语义槽位，并系统过滤、时序对齐。
5. **训练与执行基础设施：** Wan VAE/VLM 特征缓存、分层 MoT 编译、显存/吞吐优化，以及缓存视觉 K/V 的异步动作块执行。

## 关键架构与训练细节

- 视频 backbone：Wan2.2-TI2V-5B；动作专家：随机初始化 ActionDiT；语义模型：冻结 RynnBrain1.1-2B。
- 稀疏记忆：episode anchor A、近期帧 R、当前帧 C；加入 learnable Δ tokens。
- Video/action chunk：384×256、33 帧视频；动作块长度 32、80D。
- 视频/动作目标均为 flow matching；CI 两个损失分别监督相邻干净视频 latent 差分、对齐第 8 层 Δ token 与第 20 层未来视频特征。
- 4D teacher 特征预缓存；完成 235K steps 不含 4D 的预训练后，再训练 10K steps 纳入 4D 蒸馏，共 245K steps。
- 256 张 NVIDIA A800、bfloat16、有效 global batch 4096、AdamW，lr 5e-5，论文报告约 14 天。
- 推理时先运行视频 prefill 并缓存各层 K/V，再只对动作专家执行 N 步 flow denoising；没有 future-video rollout。
- 真机控制 30 Hz，动作块 32 步，每 16 个执行步异步请求新块；RTC 前缀最多 16 步。

## 数据口径

| 数据 | 处理后论文统计 |
|---|---:|
| Robot demonstrations | 11,302.20 小时 / 1,247,656 episodes |
| EgoDex + EgoVerse 输入视频 | 4,061.35 小时 / 1,622,756 episodes |
| Ego2Robot 合成机器人数据 | 5,633.77 robot-hours / 3,730,513 episodes |
| Hy-UMI-10K | 2,075.42 小时 / 416,053 个训练片段 |

Ego2Robot 的 robot-hours 是从人类视频成功转换生成的输出，不是新增原始人类采集时长。论文称 processed data 仅在上游许可允许时发布；总量不代表每一来源都可重新分发。

## 关键结果摘录

- LIBERO-Plus：92.8% success。
- RoboTwin 2.0：Clean2Clean 90.0%，Clean2Random 71.9%。
- EBench：success 49.2%，overall score 66.0。
- RoboDojo：平均 success 23.9%。
- 真机：AC-One、Arx5、Franka+XHand、TianJi Marvin+Wuji Hand 共 11 个任务。
- AC-One 烤面包：无预训练 4/20，对比预训练 19/20；Luminol 反应 0/20 对比 19/20。
- 单 RTX 5090 部署报告 30 Hz 控制；优化后 controller-observed action-chunk RTT 152.8 ms（5.11× 对比 780.5 ms standard runtime）。

## 消融

LIBERO-Plus 独立训练消融：

| 累积组件 | 成功率 |
|---|---:|
| Baseline | 49.59% |
| + SMC | 53.47% |
| + RynnBrain1.1-2B（VLM） | 69.08% |
| + Causal Imprint | 70.80% |
| + future-feature alignment | 76.45% |
| + Track4World 4D distillation | 78.37% |

注意：作者称每行均从头独立训练，不是连续微调同一个 checkpoint。

## 局限（作者自述）

论文尚未系统研究不同 ego 数据来源、Ego2Robot 转换策略、混合比例及监督类型对结果的影响；agent-assisted control 只测试有限的 corrective setting，未系统考察调用策略、长程规划或外部 agent 的紧耦合。

## 许可与引用边界

- arXiv 论文显示 CC BY-NC-SA 4.0。
- 该许可适用于论文文本，不能据此推断机器人数据、模型 backbone 或第三方数据许可。
- arXiv HTML 为内容事实来源；结果表、实验协议与详细限制以论文全文为准。
