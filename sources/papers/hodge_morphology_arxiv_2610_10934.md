# 高阶形态先验（Hodge 消息传递）论文来源归档

## 元数据

- **题目：** Higher-Order Morphology Priors for Quadruped Reinforcement Learning Under Actuator Degradation
- **作者：** Derek You, Zafir Shamsi, Keqin Wang, Christine Allen-Blanchette
- **单位：** arXiv 页面未列机构；周报标注为普林斯顿大学及两所中学（未在 arXiv 页面核实）
- **预印本：** https://arxiv.org/abs/2610.10934
- **HTML：** https://arxiv.org/html/2610.10934v1
- **版本：** v1，2026-10-07，cs.RO（交叉 cs.LG）；IROS 2026 Workshop BLPC 接收
- **代码与项目页：** arXiv 页面未列，截至 2026-10-12 未发现官方代码。

## 论文要解决的问题

执行器退化（关节最大力矩下降）时，剩余关节需要跨肢体补偿。先前工作显示形态感知的图策略对身体扰动有帮助，但图只建模成对（节点-边）连接。论文问：显式加入更高阶（面，rank-2）的机械结构，是否进一步增强整体补偿能力。

## 方法与训练信息

1. **胞腔复形：** Unitree Go1 表示为 12 个 0-cell（驱动关节）、16 个 1-cell（8 条腿内运动学边 + 4 条髋环边 + 4 条髋-小腿闭合边，图注称后者为 virtual）、5 个 2-cell（4 个肢体面 + 1 个由四个共面髋构成的机身面）；关联矩阵 B₁∈ℝ^{12×16}、B₂∈ℝ^{16×5}。
2. **Hodge 消息传递：** L_k = B_kᵀB_k + B_{k+1}B_{k+1}ᵀ；同阶消息 S_k = L_k H_k；跨阶消息 C_k = B_kᵀH_{k−1} + B_{k+1}H_{k+1}；混合 M_k = α_k S_k + (1−α_k) C_k（α_k 逐层逐阶可学习）；更新 H_k ← H_k + RMSNorm(g_k(M_k))。
3. **Actor：** 48 维观测重排为 12×20 的每关节特征；4 层传播；每关节共享的两层解码器输出 (μ,σ)，因子化 tanh-Gaussian，12 维动作。Critic 为 MuJoCo Playground 标准 MLP（123 维特权观测），各 actor 共用。
4. **任务与训练：** MuJoCo Playground 的 Go1JoystickFlatTerrain，PPO，每策略 4 亿环境步，3 个种子（11、12、13）。
5. **退化模型：** τ_i^max = ρ_i·τ_i^max,nom，ρ∈[0,1]；每次退化 2 个执行器，共 66 个无序对：44 个训练、22 个留出；训练课程每回合 30% 健康、50% 弱化（ρ~LogUniform(0.05,0.20)）、20% 失效（ρ=0）；评测在 22 个留出对与 ρ∈{0,0.025,0.05,0.075,0.10,0.15,0.20,0.25,0.50} 上平均。
6. **基线（参数量匹配）：** MLP-C（集中式 MLP）、MLP-D（共享权重的逐关节 MLP，无消息传递）、GCN（2 层）、Hodge-L（节点-边，rank 0–1）、Hodge-F（节点-边-面，rank 0–2）。

## 论文报告的结果

- **健康训练→健康测试（H→H）：** MLP-C 回报 33.88、速度 RMSE 0.052；Hodge-F 32.58、0.056；Hodge-L 32.30；MLP-D 31.17；GCN 24.52（RMSE 0.302）。生存率均为 1.00。
- **退化训练→退化测试（D→D）：** Hodge-F 回报 22.40±0.63、生存率 0.94±0.01、速度 RMSE 0.252±0.036，均为最佳；MLP-C 19.98 / 0.88 / 0.324；Hodge-L 18.61 / 0.83；MLP-D 18.15 / 0.83；GCN 15.35 / 0.73。
- **退化训练→健康测试（D→H）：** Hodge-F 回报 27.17±1.73，RMSE 0.220±0.060，最佳；MLP-C 24.85。
- **健康训练→退化测试（H→D，零样本）：** MLP-C 最优（回报 16.89、生存 0.69）；Hodge-F 15.59 / 0.60；Hodge-L 15.32 / 0.63（RMSE 0.195 最低）。即 Hodge 的优势依赖“用退化课程训练”，不能外推为零样本更强。

## 复现与解释边界

- 仅 Go1、仅平地摇杆任务、仅仿真、仅双执行器退化；3 个种子，部分标准差较大（Hodge-F D→D RMSE 0.252±0.036）。
- 论文无专门局限节；Hodge-L、GCN、MLP-D 与 Hodge-F 的差异不止“是否有面”一项，因果归因偏弱。
- 图注“virtual 边”与正文对闭合边的描述不一致。
- 无公开代码与真机实验。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-hodge-morphology-priors-quadruped-degradation.md
- 概念入口：wiki/concepts/graph-neural-network.md
- 汇总页：wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.10934>
- arXiv HTML 全文：<https://arxiv.org/html/2610.10934v1>
