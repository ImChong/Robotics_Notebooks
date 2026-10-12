# LeapBot-JAC 论文来源归档

## 元数据

- **题目：** LeapBot-JAC: Joint Action Chains for World Action Models
- **作者：** Chenhao Zhu, Baibing Liu, Pei Liu, Rui Yang, Lang Zhang, Mingyue Feng, Yaonong Wang, Jun Ma
- **单位（HTML 版作者脚注）：** 零跑汽车（Leapmotor）；香港科技大学（广州）；Jun Ma 另挂香港科技大学。arXiv 摘要页未列机构
- **预印本：** https://arxiv.org/abs/2610.12886
- **HTML：** https://arxiv.org/html/2610.12886v1
- **版本：** v1，2026-10-09，cs.RO
- **代码：** 论文给出 https://github.com/LeapWM/leapbot-jac ；截至 2026-10-12 核查，仓库只有仅含标题的 README（见 `sources/sites/leapbot-jac-github.md`），未发布代码

## 论文要解决的问题

动作分块（action chunking）策略一次预测多步控制，但只执行前一段便重新规划。在固定动作词元预算下，这些词元该如何分配，才能让执行段受益于更远的未来信息，而不增加词元或参数。

## 方法与训练信息

- 固定 32 个动作词元，在同一个共享动作专家内拆成两条链：参考链 R（16 个稀疏未来控制，步长 s=3，索引 0,3,…,45，覆盖 48 步窗口）与执行链 A（16 个稠密目标，索引 0…15）。完整窗口内 R 与 A 共享 6 个时间点，32 个词元覆盖 26 个不同控制时刻。
- 两条链都用演示控制做监督；越过 episode 末尾的索引取最后一个有效动作并标为 padding，不进直接损失但对注意力可见。
- 共享投影、时间嵌入、Transformer 层与输出头；线性 flow matching，两链共用噪声水平 σ、各自独立高斯噪声。
- 物理时间对齐：位置编码取 p=[0,3,…,45; 0,1,…,15]，在 RoPE 下执行词元 i 与参考词元 j 的相对偏移为 3j−i；沿用宿主频率表，无链身份嵌入。
- 有向注意力 M_JAC = M_host ∧ M_role：执行链可读参考链，反向不可读。
- 损失：E[ℓ_A + λ_R·ℓ_R] + 宿主世界损失，λ_R=0.2。
- 推理：R 与 A 联合从 σ=1 积分到 0，只有 A 反变换并执行；执行步数 K=10（LIBERO/LIBERO-Plus）、K=16（RoboTwin）。
- 宿主：FastWAM（复现 11,701.62M 参数）与 LeapBot-WA（复现 8,428.11M 参数）；宿主的世界建模流程、视频帧索引、世界损失系数保持不变。
- 8 张 NVIDIA H100 训练与评测；学习率、batch、步数遵循宿主原配方，所读内容中未给出；所有本地实验单一训练种子，无误差棒。

## 论文报告的结果

- **LIBERO 平均成功率：** FastWAM 97.0 → 98.3（+1.3）；LeapBot-WA 97.3 → 98.3（+1.0）。
- **LIBERO-Plus（零样本，不训练 Plus）：** FastWAM 53.5 → 70.4（+16.9）；LeapBot-WA 73.1 → 77.3（+4.2）。表中引用的 ImageWAM 83.1 为其自报平均，口径与复现行不一定等价。
- **RoboTwin 2.0 平均：** FastWAM 91.02 → 91.24（+0.22，Clean 子集 91.44 → 91.32 略降）；LeapBot-WA 91.49 → 92.00（+0.51）。文献基线 ImageWAM 93.38、LingBot-VA 92.24 训练配方与预算不同。
- **消融（FastWAM，LIBERO 平均）：** 链间可见性 R↔A 97.5、R→A 98.3（默认）、无跨链 97.7、A→R 98.2；λ_R 在 s=2 下 0.1/0.2/0.5/1.0 分别为 97.8/98.1/97.5/96.2；步长 s=2/3/4 为 98.1/98.3/98.0。
- **开销：** p50 单次模型调用延迟 FastWAM 337.0 → 337.7 ms、LeapBot-WA 438.6 → 435.1 ms；峰值显存持平（25.00 / 26.13 GB）；参数量不变。
- **真机两任务（每模型每任务 200 次试验）：** T1 顺序分拣 FastWAM 82 → 93、LeapBot-WA 90 → 95；T2 关抽屉 FastWAM 77 → 79、LeapBot-WA 85 → 90。

## 复现与解释边界

- 单训练种子、无显著性检验，论文自己把差异称为描述性。
- 步长与 λ_R 同时影响多个训练因素，消融不能干净分离原因；仅 LIBERO 的消融不能解释 LIBERO-Plus 的增益。
- 参考动作的语义未被显式约束，消融并未证明其表示抽象规划。
- 有向掩码不保证独立，经宿主世界状态或缓存可能存在间接反向通路。
- 附录读取情况：正文与附录 B 的限制性陈述已读到；训练超参未读到。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-leapbot-jac-joint-action-chains.md
- 汇总：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md

## 一手来源

- <https://arxiv.org/abs/2610.12886>
- <https://arxiv.org/html/2610.12886v1>
