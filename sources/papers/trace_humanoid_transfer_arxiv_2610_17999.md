# TRACE: Efficient Transfer Learning for Neural Whole-Body Control Across Humanoid Embodiments（arXiv:2610.17999）

> 来源归档（ingest）

- **标题：** TRACE: Efficient Transfer Learning for Neural Whole-Body Control Across Humanoid Embodiments
- **arXiv：** <https://arxiv.org/abs/2610.17999>（HTML：<https://arxiv.org/html/2610.17999>）
- **项目页：** <https://bfm-trace.github.io/>（归档：[trace-humanoid-transfer-project.md](../sites/trace-humanoid-transfer-project.md)）
- **代码（官方仓库，已核查）：** <https://github.com/QinnyuLan/SONIC_unitree_h2>（归档：[trace-sonic-unitree-h2.md](../repos/trace-sonic-unitree-h2.md)）
- **作者：** Qingyu Lan、Zhihao Tian、Zifan Wang、Junwei Liang（通讯作者）
- **机构（项目页）：** 香港科技大学（广州）；香港科技大学（Junwei Liang 兼署）
- **入库日期：** 2026-10-17
- **文章来源：** 具身智能小站 2026-10-17 公众号文章
- **一句话说明：** 把 NVIDIA SONIC 全身控制器从 Unitree G1 迁移到 H2：输入 / 输出投影注入运动学先验、对齐名义关节动力学、编码器蒸馏 + 低秩 RL，约 8 GPU 小时。
- **注意：** 与已有 [TRACE 因果记忆](../../wiki/entities/paper-trace-causal-memory.md) 同缩写不同论文，不合并。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 列出 GitHub 仓库，未写 "coming soon"，无 Hugging Face 链接。
- **仓库核查（2026-10-17）：** 仓库 `QinnyuLan/SONIC_unitree_h2` 公开（35 次提交，Apache-2.0 源码许可）；README 称 H2 检查点位于 `models/sonic_h2_released/`，提供 Sim2Sim（MuJoCo + C++ 控制器 + DDS）、Sim2Real、VR 遥操作入口。检查点文件本身、尺寸与可加载性、最近提交日期未核实；本次未本地运行。
- **结论：** **已开源（部署侧）**：H2 部署、回放、规划与遥操作代码及发布检查点说明可获取；论文的适配训练管线（LoRA-PPO 课程）是否在仓库内未核实。

## 摘录 1：方法（arXiv HTML）

- 运动学先验：由配对重定向姿态拟合线性映射，向基于肢体末端 Jacobian 的先验正则；前向映射嵌入参考与本体输入投影，动作投影 P_act = S_t⁻¹ M S_s 单独变换，目标动作裁剪到归一化范围。
- 动力学对齐：目标 PD 增益由源控制器固有频率与阻尼比初始化，按折算执行器惯量缩放；连杆与接触动力学留给微调。
- 编码器蒸馏：仅微调目标机器人的参考编码器，以 smooth-L1 对齐量化前潜变量与量化 token。
- LoRA：默认每阶段 r=32，作用于目标参考编码器、控制解码器与 critic；单阶段基线用 r=64 匹配累计预算。
- 两阶段 PPO 课程：阶段 1 均衡采样并加强全局位置 / 末端跟踪奖励，随后合并并重置 LoRA；阶段 2 自适应优先高误差运动。
- 数据：从 BONES-SEED 按类别均衡与命令空间选择 10M 帧；评测用 67,953 条未参与训练的 PHUMA 运动。
- 算力：1 张 RTX PRO 6000 约 8 小时，12,288 并行环境，称低于 SONIC 21,000 GPU 小时的 0.1%（不含源预训练，GPU 小时未跨硬件归一化）。

**对 wiki 的映射：** [`paper-trace-humanoid-transfer.md`](../../wiki/entities/paper-trace-humanoid-transfer.md)；与 [NVIDIA SONIC](../../wiki/entities/nvidia-sonic.md)、[LoRA](../../wiki/concepts/lora.md)、[全身控制](../../wiki/concepts/whole-body-control.md) 互链。

## 摘录 2：结果

- PHUMA 67,953 运动：H2 上 TRACE 96.08%，G1 上 SONIC 97.59%；去掉预适配 87.29%，去掉数据策略 93.80%，去掉两阶段微调 95.64%；两阶段课程比单阶段降低根位置误差 26.2%。误差阈值 G1 0.250 m、H2 0.3447 m（按身体尺寸调整）。
- Direct Finetune 收敛更慢；Scratch 在 3k 迭代内未得到可用跟踪器。
- 真机：VR 全身遥操作（开门、装玩具、丢垃圾、搬椅子、踢球）；用遥操作示教后训练 GR00T-N1.7，五个 VLA 任务各 20 次：搬玩具（单臂）90%，搬方块（双臂）75%，拿枕头 80%，开门 65%，开垃圾桶 65%。
- 局限（作者）：仅一对本体与一个预训练 BFM；无可复用 critic 的高效适配未探索；计划加入 RGB、深度与触觉。

**对 wiki 的映射：** 评测、结论、局限小节。
