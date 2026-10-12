# LeWAM 论文来源归档

## 元数据

- **题目：** LeWAM: End-to-End World and Action Modeling with JEPAs
- **方法：** LeWAM（JEPA 潜在动力学 + 动作流匹配 + SIGReg，联合端到端训练）
- **作者：** Minghao Fu*, Tavis Siebert*, Eryk Halicki, Randall Balestriero
- **单位：** UC San Diego；ETH Zurich；University of British Columbia；Brown University（页面未逐人对应）
- **预印本：** https://arxiv.org/abs/2610.12818
- **HTML：** https://arxiv.org/html/2610.12818v1
- **版本：** v1，2026-10-08，cs.RO
- **项目页：** https://le-wam.github.io/（归档见 [le-wam-github-io.md](../sites/le-wam-github-io.md)）
- **代码：** https://github.com/MinghaoFu/lewam（归档见 [lewam.md](../repos/lewam.md)）
- **入库日期：** 2026-10-12
- **读取范围：** arXiv HTML 前 100,000 字符（含正文与附录 A.2 之前）；其后 50,036 字符（附录 B–F 及部分 C 表）**未读到**。

## 论文要解决的问题

JEPA 无需像素重建即可学潜在世界模型，但常规流程先训表示与动力学、再接策略或求解器。作者指出：JEPA 特征未必保留控制相关信息（可能偏好时间上持久的干扰因素）；以宽高斯先验的 CEM 规划，在动作维度或时域增大时需要大量 rollout，接触转换又使局部动作优化困难。

## 方法与训练信息

1. **编码器：** ResNet-18（从零训练，GroupNorm 16 组）+ 空间 softmax（32 关键点）+ 2 层投影 MLP（BatchNorm）；潜变量 192 维，与 LeWM 对齐；输入 224×224 裁到 202×202。
2. **预测器（Mixture-of-Transformers）：** 状态分支 f_z 预测 z_{t+K}；动作分支 f_a 预测带噪动作块的流速度（条件含流时间 s、历史与可选目标）。两分支各自拥有 Q/K/V、输出、归一化与前馈权重，共享一个带掩码的自注意力；掩码阻止干净动作泄漏进动作分支。宽度 192、4 块、4 头、dropout 0.1，目标条件用 AdaLN-Zero；约 16.9M 可训练参数。
3. **损失：** L = L_dyn（下一潜变量 MSE）+ L_act（流速度 MSE）+ λ_sig·L_sig（SIGReg：Epps–Pulley 统计量在 M 个随机单位投影上平均，把潜变量推向各向同性高斯），λ_sig = 0.04。流匹配 A^(s) = (1−s)ε + sA，目标速度 A−ε。
4. **时间参数：** 帧跳 K=5，动作块 C=10（=2K）；视觉规划历史步长 Δ=K。
5. **规划（非 CEM）：** 从学到的流策略采样 N 个候选（8 步 Euler）→ 经潜在动力学前向 L 步 → 对冻结动力学做梯度下降（目标代价 MSE(ẑ_{t+L}, z_g)，对采样计划加二次邻近惩罚；Adam lr 0.05、梯度裁剪 10、a_max=3、λ_prox=0.01）→ 取最低代价迭代，执行前 K 个动作，滚动重规划。迭代次数 J 在已读文本中未给出。
6. **训练：** 视觉规划 batch 128、50 epoch、基础 lr 1.5e-4、warmup 5 epoch、AdamW（权重衰减 1e-4）、梯度裁剪 1、种子 42；无目标任务完成 batch 64、120 epoch、lr 1e-4。余弦调度，90/10 划分，检查点按验证集动作流匹配损失选择而非任务成功率。

## 论文报告的结果

- **十个仿真环境（专家/随机/混合质量数据）：** 四个改编自 LeWM 基准，六个为新增接触丰富操作与导航任务（Push-T、Cube、Scene、Puzzle、Two-Room、ToolHang、Transport、DrawerCleanup 等）。视觉规划协议：目标偏移 L=25，预算 50 动作，3 种子（LeWM CEM 为 10 种子）。
- **视觉规划：** LeWAM 平均 93.5%，LeWM(CEM) 50.9%；ToolHang 94.7% vs 35.4%；Transport 83.3% vs 2.8%；Puzzle 80.7% vs 44.0%（GCBC）。摘要口径：接触丰富任务 28.6%→89.7%，长时域 10.9%→46.2%（100 步目标偏移；Push-T/Cube/Scene/Puzzle）；规划速度 52.50 vs 1.62 plans/s，32.3×。（摘要的 89.7% 与十环境均值 93.5% 为不同子集口径。）
- **线性探针（Table 1，R²）：** LeWAM 含 SIGReg：有效秩 42.90、物体 0.90、本体 0.86；去掉 SIGReg 有效秩降为 10.94；LeWM 含 SIGReg 有效秩 94.20、物体 0.95、本体 0.80。
- **消融：** 去 SIGReg 使 Push-T 规划降 25.3 点、Puzzle 5.3、Scene 4.0；端到端编码器直接成功率 86.7%，冻结 DINOv2 CLS 特征 79.0%，冻结 patch 特征 75.8%；流匹配的规划增益大于 MSE 头+噪声注入（具体数字仅在图中）。
- **无目标任务完成（仿真）：** 与 Diffusion Policy 在两个任务上相当，Transport 上高 16 点（详表在未读附录）。
- **真机（Table 2，每任务 30 次）：** Diffusion Policy 19/16/0（平均 38.9%，270.0 GFLOPs/chunk）；π0.5 14/3/9（28.9%，3603.6 GFLOPs）；LeWAM(冻结 V-JEPA 2.1) 15/12/5（35.6%，270.2）；LeWAM 16/8/6（33.3%，270.2）。三项为 Pick and Place T、Push-T、Tape Stack。LeWAM 平均低于 Diffusion Policy，高于 π0.5，约 13.3× 更少运算。

## 复现与解释边界

- 作者自述限制：数据规模与覆盖有限；未验证对陌生物体/视角/接触状态的泛化；每任务、每本体单独训练，无多任务或语言条件；未与层次/子目标规划结合；模型仅约 17M 参数。
- 真机结果是“有竞争力”而非优于 Diffusion Policy。
- 项目页写 arXiv “coming soon”，实际 arXiv 已上线。
- 附录部分（含无目标任务详表）未读到。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-lewam.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md
- 周报来源：[wechat_embodied_station_34_papers_spatial_grounding_2026-10-12.md](../blogs/wechat_embodied_station_34_papers_spatial_grounding_2026-10-12.md)

## 一手来源

- <https://arxiv.org/abs/2610.12818>
- <https://arxiv.org/html/2610.12818v1>
- <https://le-wam.github.io/>
- <https://github.com/MinghaoFu/lewam>
