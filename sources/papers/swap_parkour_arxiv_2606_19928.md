# SWAP: Symmetric Equivariant World-Model for Agile Robot Parkour（arXiv:2606.19928）

> 来源归档（ingest）

- **标题：** SWAP: Symmetric Equivariant World-Model for Agile Robot Parkour
- **类型：** paper / 四足 locomotion / 感知跑酷 / 潜变量世界模型 / 对称等变网络
- **arXiv：** <https://arxiv.org/abs/2606.19928>（PDF：<https://arxiv.org/pdf/2606.19928>）
- **项目页：** <https://swap-parkour.github.io/>
- **作者：** Kaixin Lan*, Ze Wang, Hongyi Li, Lei Jiang, Chaojie Fu, Chengkai Su, Choi Lam Wong, Yongbin Jin†, Hongtao Wang†（浙江大学 X-Mechanics、ZJU-Hangzhou 科创中心、Mirrorme Technology）
- **入库日期：** 2026-06-19
- **一句话说明：** 在 **RSSM 潜变量世界模型 + Actor-Critic** 上 **硬约束左右镜像等变**（SE-CNN/SE-MLP/SE-GRU），端到端联合训练四足 **极限跑酷** 策略；**Apollo 四足** 实机 **2.13 m 远跳 / 1.63 m 攀台**（迄今四足跑酷纪录级），并对 **镜像地形** 与多样户外场景 **零样本泛化**。

## 摘要级要点

- **问题：** 现有 latent world model（Dreamer / DayDreamer / WMP 系）纯数据驱动，把左右对称物理交互当作独立模式重复编码，增大学习负担、削弱几何正则，限制下游策略对潜空间的利用效率。
- **主张：** 将 **对称等变** 直接嵌入 **世界模型与 Actor-Critic**——镜像观测 → 镜像潜状态；**等变 Actor** 输出镜像动作；**不变 Critic** 对镜像状态给出一致价值估计。
- **框架：** **Symmetric MDP（SMDP）** 形式化四足形态与环境左右对称；**低频 Symmetric Equivariant RSSM**（每 $k$ 步更新潜状态）+ **高频 Equivariant Actor / Invariant Critic**（以 `sg(h_t)` 为条件）；图像编解码用 **$\mathbb{Z}_2$ 反射群 SE-CNN**，其余用 **SE-MLP / SE-GRU**。
- **辅助重建：** 除深度图与本体 proprio 外，解码 **body/foot-centric heightmap** 作为辅助目标，迫使潜空间压缩地形几何、加速收敛。
- **训练：** Isaac Gym **6000** 并行环境 + NVIDIA Warp；**课程学习** 逐步放开地形难度与目标速度（上限 **3.0 m/s**）；**对称等变 AMP** 判别器；单卡 RTX 4090 约 **10 h** 端到端收敛；策略 **零样本** 部署真机。
- **消融：** SWAP（全等变）> SWAP w/o Eq-Policy（仅 WM 等变）> SymLoss（软约束）> SWAP w/o Eq（≈ **WMP** 无等变基线）；**镜像 OOD 地形** 上无等变基线重建误差陡升、成功率骤降。
- **极限跑酷（仿真课）：** 统一课程含最高 **1.9 m** 箱台、最宽 **3.0 m** 沟壑，及楼梯、**60°** 侧倾坡、不规则石堆；**1.0 m/s** 指令下 gap / box 成功率 SWAP 最优。
- **实机纪录（Table IV，Apollo 70×55 cm、72 kg）：** **213 cm** 远跳、**163 cm** 攀台——论文称在对比系统中为 **最远 / 最高** 四足跑酷绝对指标；相对体尺归一化约 **3.0×**（与 PIE 峰值同级）。
- **户外零样本：** 单一策略直接部署湿地反光、暗楼梯、高草/纸板非刚性障碍、碎石与可移动地毯等，无需环境微调。

## 对 wiki 的映射

- 沉淀实体页：`wiki/entities/paper-swap-parkour.md`
- 交叉更新：`wiki/tasks/locomotion.md`、`wiki/tasks/stair-obstacle-perceptive-locomotion.md`、`wiki/entities/extreme-parkour.md`
- 姊妹路线：Extreme Parkour（MFRL + 蒸馏）、WMP（无等变 RSSM 跑酷）、PIE/START（短程时序）、DreamWaQ++（隐式地形想象）
