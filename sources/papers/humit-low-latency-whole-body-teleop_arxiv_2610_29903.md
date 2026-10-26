# HuMiT（arXiv:2610.29903）

> 来源归档（paper）

- **标题：** HuMiT: Low-Latency Whole-Body Humanoid Teleoperation via Compact Reference Tracking
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.29903>
- **PDF：** <https://arxiv.org/pdf/2610.29903>
- **作者：** Liu Cao, Botian Xu, Ruoqu Chen, Jiajun Liu, Mingzhi Pei, Zihao Wang, Zhongyu Li, Mengdi Xu
- **arXiv 评论栏：** 7 pages, 3 figures, 4 tables. Project page: this https URL
- **外链：** <https://xiaohu-art.github.io/Track/>
- **入库日期：** 2026-10-26
- **一句话说明：** HuMiT：基于紧凑参考跟踪的低延迟人形全身遥操作。

## 开源状态

- arXiv 摘要页/评论给出项目页或外链（https://xiaohu-art.github.io/Track/）；未发现代码仓库链接（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】人形／四足39篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
2. **周报所述问题：** 密集全身参考、未来帧条件和在线动作重定向会显著增加遥操作延迟。
3. **周报所述创新点：** 控制器只接收根节点高度与速度，以及双腕、双踝四个关键点的当前 SE(3) 位姿；无需在线优化和未来参考，通信到策略输出延迟约 20 毫秒。
4. **arXiv 摘要（英文节选）：** Real-time whole-body humanoid teleoperation remains challenging due to the computational latency introduced by optimization-based retargeting pipelines and future-frame conditioning. Existing motion tracking policies typically require dense reference targets including full joint positions, body-part transformations, and even future frames, which tightly couples the controller to expensive upstream modules. In this work, we present HuMiT, a whole-body teleoperation system built on a unified tracking policy that requires only a compact reference target, consisting of root height, root velocity, and the SE(3) poses of four sparse keypoints at the current frame, yet achieves competitive tracking fidelity compared to methods relying on more diverse reference states including complete joint positions, dense body-part transformations, and even future frames.

**对 wiki 的映射**

- [paper-humit-low-latency-whole-body-teleop](../../wiki/entities/paper-humit-low-latency-whole-body-teleop.md)
