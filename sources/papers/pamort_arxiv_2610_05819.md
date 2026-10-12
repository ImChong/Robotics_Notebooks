# PAMORT 论文来源归档

## 元数据

- **题目：** Transporting Unsecured Stacked Payloads with a Quadrupedal Robot via Multi-Objective Reinforcement Learning
- **方法：** PAMORT（多目标基础策略 + 偏好权重调节器）
- **作者：** Nobuo Namura, Masayuki Hiromoto, Kento Uemura, Hironobu Sasaki, Kanata Suzuki
- **单位：** Fujitsu Limited（日本川崎）
- **预印本：** https://arxiv.org/abs/2610.05819
- **HTML：** https://arxiv.org/html/2610.05819v1
- **版本：** v1，2026-10-05，cs.RO（交叉 cs.LG），8 页 6 图
- **代码与数据：** arXiv 页与正文均未列项目页或 GitHub；截至 2026-10-12 未发布。

## 论文要解决的问题

腿式机器人在不平地形上背负未固定载荷时，机器人自身不摔倒并不保证载荷不滑落或倾倒：激进运动能提高速度跟踪，却会让堆叠箱体倒下。固定权重把运动奖励与载荷稳定奖励合成单一标量，难以随地形与堆叠情况改变取舍。实验平台为 Unitree Go2，躯干上固定一块板，板上堆放箱子，没有载荷传感器和主动夹持。

## 方法与训练信息

1. **偏好向量：** 二维 w = [w_move, w_payload]，位于 1-单纯形；训练时每个 episode 从 Dirichlet(1,1) 采样。
2. **奖励分组：** 17 项 move 项（速度跟踪、关节与动作正则、步态、不期望接触）与 6 项 payload 项（底/顶箱姿态对齐、底/顶箱平面加速度、底箱掉落 −150、顶箱掉落 −100）。
3. **基础策略：** 以 TAR 框架为骨干，PPO + 非对称 actor-critic；actor 输入本体感知、64 维学生潜变量、3 维速度估计与 w，输出 12 维关节位置目标；actor MLP [512,256,128]（ELU），critic 为每个目标一个价值头。学生为 1 层 LSTM（隐层 256），教师为 MLP [256,128,64] 读取 195 维特权观测。
4. **多目标 PPO：** 各目标组的优势分别标准化，再按 Ã = wᵀA 合成。
5. **偏好感知课程：** 以偏好加权几何平均计算能力（c_min = 0.05）；聚合能力 ≥0.8 升级地形、≤0.5 或摔倒则降级。
6. **权重调节器（WA）：** 冻结基础策略，输入 o_t、潜变量、速度估计与上一时刻 w；输出 logits 经 softmax；每 4 个低层步（0.08 s）更新一次；平滑项 λ_t‖w_t − w_{t−1}‖²（λ_t∈[0.02,0.5]）；训练 5,000 次迭代。
7. **仿真与规模：** Isaac Lab；多目标基础策略 16,384 环境 / 8 GPU，TAR* 与 WA 为 8,192 环境 / 4 GPU；控制 50 Hz；所有策略 130k 迭代；PPO 24 步/迭代、5 epoch、4 mini-batch、Adam 1e-4（WA 5e-4）。训练时长原文未报告。

## 论文报告的结果

- **仿真（地形等级 0–9 平均成功率，TAR* vs PAMORT）：** 0 个载荷 0.782/0.801；1 个 0.723/0.758；2 个（训练配置）0.650/0.687；3 个（未见）0.530/0.550。PAMORT 在 SlopeDown 与 NarrowStairsDown（两载荷）略逊于 TAR*。
- **与逐地形最优固定偏好的 oracle 对比：** 平均 0.699 vs 0.738（94.7%）；三载荷 0.550 vs 0.570（96.5%）。
- **真机 8 个任务（38%/44% 坡、16/20 cm 台阶 × 两/三载荷，每任务 10 次）：** 两载荷均值 0.875 vs 0.650；三载荷 0.825 vs 0.700；总体 0.850 vs 0.675。20 cm 台阶上 PAMORT 并不占优（两载荷 0.6 vs 0.6，三载荷 0.7 vs 0.8）。
- **推理耗时：** TAR* 0.4 ms、PAMORT（含 WA）0.5 ms，对比 20 ms 控制周期。
- **预设偏好分析：** w_payload < 0.4 时常掉载荷。

## 复现与解释边界

- 无公开代码与权重；超参已较完整，但载荷物理参数、地形生成细节仍需自行实现。
- 载荷仅为刚性箱体，仅两个目标组；多目标扩展为开放问题。
- 真机每任务 10 次、地形为木质托盘，统计置信有限；基线 TAR* 为作者自行加入载荷项并用固定权重的版本。
- 单位：论文机构为 Fujitsu Limited；与同周 2610.08637 的 Fujitsu Research of America 是不同署名单位。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-pamort-stacked-payload-quadruped-morl.md
- 周报汇总：wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.05819>
- arXiv HTML 全文：<https://arxiv.org/html/2610.05819v1>
