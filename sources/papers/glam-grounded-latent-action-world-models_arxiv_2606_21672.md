# GLAM：Imitation from Heterogeneous Demonstrations using Grounded Latent-Action World Models（arXiv:2606.21672）

> 来源归档（paper）

- **标题：** Imitation from Heterogeneous Demonstrations using Grounded Latent-Action World Models
- **作者：** Tianyou Wang, Anson Lei, Joe Watson, Ingmar Posner（University of Oxford）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.21672>（v1，2026-06-19，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2606.21672v1>
- **项目页：** <https://viccccciv.github.io/glam/>（论文称视频与代码在此；本次未访问，未确认是否有 GitHub 仓库）
- **入库日期：** 2026-11-12
- **一句话说明：** 用一对共享连续潜动作空间的生成模型（IDM 后验 + 动作编码器后验，非对称 KL 对齐）把带动作标签的目标机器人数据与无动作标签的辅助数据统一到同一动作表示，再用该潜动作监督下游 MIP 行为克隆策略。
- **核对范围：** 下载 arXiv v1 PDF（17 页），pdftotext 逐段读完正文第 1–5 节（含摘要、相关工作、方法、实验、结论与局限）与附录 A–E（跨源迁移、平滑度分析、第三数据源、实验设置、实现细节）；图 3、图 4 看了渲染图，图 3 柱高仅作大致读数，具体数值以正文为准；参考文献只扫读未逐条核对；未访问项目页与任何代码，未复现。
- **通读：** 已完成

## 核心摘录

1. **问题设定（第 3 节）：** 目标域 D_tar 有图像、末端位姿、本体状态、动作；辅助域 D_aux 只有观测轨迹，无动作标签也无机器人状态。
2. **方法（第 3.1 节，式 1–6）：** 异质模型（只用观测，Dreamer 式潜动力学加 IDM 后验 q_φ(z|x_t,x_{t+1})）在 D_tar ∪ D_aux 上联合训练；目标模型含动作编码器 q_η(z|s,a)、本体前向模型与动作解码器，仅在 D_tar 上训练；式 5 用 stop-gradient 的非对称 KL 把 IDM 后验拉向动作编码器后验；总损失 -J_H - J_T + λ·C_KL。
3. **下游 BC（第 3.2 节，式 7–8）：** 用 IDM 后验均值给所有转移重标潜动作；DINOv2-S/14 双视角编码加 MIP 策略预测长度 H=20 的潜动作块，再由动作解码器译成机器人动作；监督损失分为全部数据的潜动作回归与仅目标数据的动作回归。
4. **GLAM-O 变体（第 3.1 节）：** 世界模型输入由 RGB 换成被操作物体的二值分割掩码（现成分割模型），目标函数不变。
5. **实验设置（第 4.1 节、附录 D）：** 5 个任务（真机 3：lifting、pick-and-place、knock-down；仿真 2：单臂 stack-two、双臂 stack-three），D_tar 为 100 条目标机器人轨迹，D_aux 为 400 条仿真中的 UMI 夹爪轨迹（脚本策略生成，自由浮动末端，无关节动作）。真机目标平台 Kinova Gen3 加两台 RealSense。真机每任务 20 次试验，仿真 3 个训练种子各 50 次试验。
6. **主结果（第 4.3 节、图 3）：** GLAM-O 相对最强基线，真机三任务平均 +35%，仿真 stack-two +44%，双臂 stack-three +69%；stack-three 上 GLAM(-O) 为 72.7%，所有基线 ≤4%；摘要称相同数据稀缺设定下平均提升 +48%（正文未给出该平均的计算口径）。
7. **数据效率（图 4）：** stack-two 上 MIP 需约 5 倍目标轨迹才达到 GLAM-O 的成功率；在 100 条目标基础上，加辅助 UMI 数据与加目标数据的提升曲线基本重合（总量到 500 条）。
8. **跨源迁移（第 4.2 节、图 2、图 5）：** UMI、Kinova-sim、Kinova-real 的未见 episode 经 IDM 取潜动作，由只在 Kinova-sim 数据上训练的动作解码器译码，在仿真 Kinova 上开环回放，定性可复现原动作。这是定性展示，无量化指标。
9. **消融观察（附录 A，定性）：** 去掉非对称对齐，两条后验互相拉近，UMI 潜动作无法迁移到 Kinova；把动作重建项挂在 IDM 而非动作编码器上，同样使 UMI 潜动作迁移失败。论文没有给出这两项消融的定量表。
10. **第三数据源（附录 C、表 1）：** 再加 100 条 Kinova-sim 轨迹，pick-and-place 15/20 到 18/20，knock-down 17/20 到 20/20，lifting 18/20 持平。
11. **作者自述局限（第 5 节）：** 辅助数据与部署共享相机位置与桌面场景，未测真实野外数据（如网络视频）；未测形态差异大的末端（灵巧手）；辅助集与目标集共享物体与任务语义；潜空间仅基于视觉物体运动、末端位姿与本体，不含接触力或触觉。

## 对 wiki 的映射

- [paper-glam-grounded-latent-action-world-models](../../wiki/entities/paper-glam-grounded-latent-action-world-models.md)
