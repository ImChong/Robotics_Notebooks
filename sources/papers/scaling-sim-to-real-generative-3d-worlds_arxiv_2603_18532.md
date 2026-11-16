# Scaling Sim-to-Real Reinforcement Learning for Robot VLAs with Generative 3D Worlds（arXiv:2603.18532）

> 来源归档（paper，全文通读）

- **标题：** Scaling Sim-to-Real Reinforcement Learning for Robot VLAs with Generative 3D Worlds
- **作者：** Andrew Choi, Xinjie Wang, Zhizhong Su, Wei Xu（Horizon Robotics）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2603.18532>（读的是 v1，2026-03-19）
- **PDF：** <https://arxiv.org/pdf/2603.18532v1>
- **代码：** 论文未给出本工作的仓库；附录 D 只提到使用第三方 π0 实现 open-pi-zero（https://github.com/allenzren/open-pi-zero）及 EmbodiedGen、ManiSkill 3
- **入库日期：** 2026-11-12
- **一句话说明：** 用语言驱动的场景设计器加 EmbodiedGen 生成 100 个可交互 3D 场景，在 ManiSkill 3 中用 PPOFlow 对 π0 做稀疏奖励 RL 微调，并做 sim-to-real 部署与场景数量消融。
- **核对范围：** 读了 v1 PDF 全部正文（摘要、§1 引言、§2 相关工作、§3 方法、§4 实验、§5 结论与局限）与附录 A 训练细节 / 奖励 / 域随机化表、B 的 K 消融、C 生成流水线统计、D 基础模型候选、E/F 场景说明与图注、G/H 的图注与定性案例文字；Table 1、Table 2 的数值逐项对照过。F.1/F.2 的逐场景指令表与 F.4–F.7 的逐场景成功率柱状图只看了结构，没有逐条核对；图中的像素细节未逐一判读。
- **通读：** 已完成

## 核心摘录（均来自论文）

1. 流程（Fig.1–3, §3）：真机数据训练模仿策略 π_pre（π0，BridgeV2 预训练，第三方 open-pi-zero 检查点）；GPT-4o 场景设计器把任务描述转成 scene graph（背景 / 上下文 / 干扰物 / 操作目标 / 机器人，关系如 ON、IN），GPT-4o 质检循环剔除物理不合理配置；EmbodiedGen 生成交互式 3D 场景；ManiSkill 3 并行仿真。
2. PPOFlow（§3, Eq.3–7）：在 ReinFlow 思路上给流匹配积分每步注入可学噪声 σ_ϕ，得到可算的联合对数概率；加价值头 V_ψ；重要性比用幂 s（0.2）缩放后进入 PPO 裁剪目标，优势用 GAE。
3. 设置（§4, 附录 A）：任务只有拾取放置；Interbotix WidowX 250S + 单个 Logitech C922 相机；192 环境，batch 19200，mini-batch 1920，episode 长度 25，γ=0.99，控制频率 5 Hz，动作块 C=4，RL 时 K=1；8 张 RTX 6000 Ada 训练 5 天；LoRA（rank 32）微调 VLM，动作专家全量微调；奖励为稀疏成功（Eq.8，物体 A 与目标 B 接触且不与桌面 / 机器人接触）。
4. 仿真结果（Table 1）：N=0（π_pre）在 EG 全部 100 场景成功率 9.7%，N=100 为 79.8%，完成时间约 10 s 降到 8 s；N 从 1 增到 50，训练场景成功率 94.3%→78.9%，EG 未见场景 53.2%→77.9%，SimplerEnv（SE）三场景 36.1%→68.4%；N=100 在 SE 上 74.3%；只在 3 个 SE 场景上训练的策略在 SE 上 96.7%，在 EG 未见场景上 36.5%。
5. 真机结果（Table 2）：12 个场景各 10 次、π_pre 与 N=100 各 120 次共 240 次；部分成功 45%→88.3%，任务成功 21.7%→75%，动力学失败 66.7%→18.3%，语义失败 18.3%→6.7%，平均完成时间 11.5 s→10.2 s。
6. 其他（附录）：A 冻结 VLM 会降低上限并在长训后塌缩；去掉 s 缩放训练会立刻崩；B 中 K∈{1,2,4} 在 W 上单次运行成功率 79.75 / 77.23 / 77.29%（K=1 最高），推理延迟 K=10 到 K=1 为 2.72×（常规）/ 2.36×（torch.compile）；C 中 100 场景含 516 个唯一物体资产，人工检查 85% 环境可直接用于 RL，单场景完全在线生成 46.8 ± 5.0 分钟，用现成资产库约 2 分钟；D 选 π0 而非 OpenVLA（real-to-sim 差距大）与 SpatialVLA（延迟高）。

## 论文自述的局限

- §5：实验只做拾取放置，因 EmbodiedGen 目前可可靠生成的物体与交互类型有限；计划扩展到铰接物体、工具使用、多阶段任务。
- §4：训练成功率随 N 增大而降，作者解释为 batch 固定使每个场景样本变少（1/N），假设同比放大 batch 可缓解，未验证。
- 附录 B：K 消融为单次运行，差距约 2–2.5%。
- 附录 C：15% 的生成环境需人工微调（尺度不匹配、初始摆放不佳）。

## 对 wiki 的映射

- [paper-sa-2603-18532-scaling-sim-to-real-reinforcement-learning-for-r](../../wiki/entities/paper-sa-2603-18532-scaling-sim-to-real-reinforcement-learning-for-r.md)
- 清单索引级摘录（旧）：[sun_awesome_r2s2r_2603_18532_scaling-sim-to-real-reinforcement-learni.md](sun_awesome_r2s2r_2603_18532_scaling-sim-to-real-reinforcement-learni.md)
