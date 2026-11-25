# Xbotics 第10讲：模仿学习实战 / Imitation Learning in Practice: ACT, Diffusion and Flow Matching

- **类型：** course / tutorial
- **作者：** Xbotics 具身智能实验室
- **发布日期：** 2026-11-24（公众号）
- **原文：** https://mp.weixin.qq.com/s/TGbxENCrF4j2SVzyACeXEQ
- **官方文稿：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/docs/part3-end-to-end/10-%E6%A8%A1%E4%BB%BF%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98.md
- **版本：** Handbook commit 342247fb71dd0ee15386b6eab80543c50593d3c1；LeRobot v0.5.1
- **代码：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook
- **入库日期：** 2026-11-24
- **核对范围：** 用户提供的第10讲完整正文（§1–6、参考文献）及官方 Markdown 章节；静态读取四个 ACT 实验脚本、LeRobot v0.5.1 的 configuration_act.py、modeling_act.py、processor_act.py。通读标记仅针对第10讲，不代表其余20讲或第三方论文全文已读；未实际训练、评测或上真机。
- **通读：** 已完成

- **官方在线书稿：** https://xbotics-embodied-ai-club.github.io/Xbotics-Embodied-AI-Handbook/part3-end-to-end/10-%E6%A8%A1%E4%BB%BF%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98.html

## 核心摘录（归纳）

- §1–2：BC 的点回归在多峰条件分布下可能落在无效模式之间；动作分块与分布建模是正交设计。ACT 训练时用真实动作块和本体状态推断 CVAE latent，推理跳过 VAE encoder、固定 z=0。
- §2–3：LeRobot 默认 chunk_size=100、n_action_steps=100、latent_dim=32、KL 权重 10、主 encoder 4 层、decoder 1 层、VAE encoder 4 层；temporal ensembling 默认关闭。启用时每步重新预测，正系数偏向较老预测。源码的 masked L1 是全张量均值，并非除以有效元素数。
- §3：LIBERO bowl-to-plate 单任务实验；SO-101 将 498 集仿真与 300 集真机合并成 798 集、273889 帧，50000 步、batch 64。作者报告同一仿真场景独立 50 局成功率 76%，不是混训提升幅度或真机成功率。
- §4：DDPM 预测噪声，条件动作序列从噪声迭代去噪，再执行一部分并滚动重规划；DDIM 可减少网络调用。
- §5–6：线性条件 Flow Matching 以噪声到真实动作的端点差作速度监督，采样用 ODE；独立端点配对不等于最优传输，边际流也不保证直线。比较 NFE、端到端时延和闭环成功率，不按算法名直接排名。

## 开源入口

- [LIBERO 训练](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/vla/3_imitation_learning/3_1_act/train_act_libero.py) 与 [闭环推理](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/vla/3_imitation_learning/3_1_act/infer_act_libero.py)
- [SO-101 混合训练](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/vla/3_imitation_learning/3_1_act/train_act_so101_sim.sh) 与 [50 局评估](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/vla/3_imitation_learning/3_1_act/eval_act_so101_sim.sh)
- [LeRobot ACT v0.5.1](https://github.com/huggingface/lerobot/tree/v0.5.1/src/lerobot/policies/act)
- 仿真器：https://github.com/Xbotics-Embodied-AI-club/Xbotics-SO101-Sim
- 课程使用的 LeRobot 分支：https://github.com/Xbotics-Embodied-AI-club/lerobot
- 仿真数据及产线（文稿列出）：https://huggingface.co/datasets/Harrysunshine/so101-sim-pickplace-v2

## 参考文献（课程所引，非本次独立论文 ingest）

- ACT：https://arxiv.org/abs/2304.13705
- LeRobot tutorial：https://arxiv.org/abs/2510.12403
- DDPM：https://arxiv.org/abs/2006.11239
- Diffusion Policy：https://arxiv.org/abs/2303.04137
- DDIM：https://arxiv.org/abs/2010.02502
- Flow Matching：https://arxiv.org/abs/2210.02747
- π₀：https://arxiv.org/abs/2410.24164

## 对 wiki 的映射


- [第10讲独立知识节点](../../wiki/entities/xbotics-handbook-lecture-10.md)
- [课程主节点](../../wiki/entities/xbotics-embodied-ai-handbook.md)
