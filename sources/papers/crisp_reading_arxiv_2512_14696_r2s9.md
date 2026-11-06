# CRISP 论文通读（Contact-guided Real2Sim）

> 来源归档（原文通读 · 支撑 `wiki/methods/crisp-real2sim.md`）

- **标题：** Contact-guided Real2Sim from Monocular Video with Planar Scene Primitives
- **作者：** Zihan Wang, Jiashun Wang, Jeff Tan, Yiwen Zhao, Jessica Hodgins, Shubham Tulsiani, Deva Ramanan（CMU）
- **类型：** paper / real2sim / human-scene-interaction
- **原文：** <https://arxiv.org/abs/2512.14696>（2026-11 通过 arXiv PDF v3 获取全文，ICLR 2026）
- **一句话说明：** 对点云做法线 K-means + DBSCAN + 光流时序合并，拟合约 50 个凸平面长方体；用 VLM 接触预测补全被遮挡支撑面；用 DeepMimic/MaskedMimic 式 RL 跟踪策略验证“可仿真”。
- **核对范围：** 通读正文 §1–§5（方法 §3.1–3.4、实验 §4.1–4.4）、附录 B–H（平面拟合算法、接触可靠性、局限、运行时间、场景感知策略、RL 超参）；图 6/7 学习曲线仅看图注。参考文献列表未逐篇核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **初始化（§3.1）：** MegaSAM 恢复内参、相机位姿与深度（优化阶段把深度估计器换成 MoGe）；GVHMR 给出 SMPL，经相机位姿提升到世界系；用人体尺度把尺度不定点云校正为米制。
- **平面拟合（§3.2、附录 B、图 3）：** 把场景分解为约 50 个凸原语；逐帧对法线图 K-means，段内 DBSCAN 空间切分，再用光流对应与法线余弦跨帧合并；每个合并区域 RANSAC 拟合平面，取最小面积矩形定轴，默认厚度 0.05 m 成长方体。
- **接触补全（§3.3、附录 C）：** InteractVLM 预测 SMPL 顶点接触；近接触帧易误报，故做时间–运动学过滤（连续 L 帧高置信且身体速度低，取最静止帧）；在预测接触点上再拟合平面补全座面、台阶等被遮挡结构（厚度下限 0.05 m）。
- **物理跟踪（§3.4、附录 H）：** 沿用 DeepMimic 的 RSI（10% 从首帧，其余均匀）与 ET（关节偏离参考 >0.5 m 终止）；策略为 Transformer 编码器，PD 目标动作，固定对角协方差 σ=0.055；Isaac Gym 120 Hz 仿真、30 Hz 策略、PPO+GAE；2048 并行环境、学习率 2e-5、γ=0.99、λ=0.95；每个动作片段单独训练策略。
- **评测设置（§4.1）：** EMDB-2（21 序列）与 PROX（12 个室内场景，约 600 帧/视频）；成功率指关节全程与参考偏差不超过 0.5 m；FPS 含仿真与策略学习。
- **主结果（表 1）：** PROX+EMDB 上 VideoMimic RL 成功率 44.8% / 16K FPS，CRISP 平面版 93.1% / 23K FPS；PROX 成功率 27.3% → 90.9%，EMDB 50.0% → 93.8%；TSDF 版 75.9%、NKSR 版 79.3%。平面版双向 Chamfer（0.187）略差于 NKSR（0.163），但单向 Recon→GT 最低（0.174）、Non-Pene 最高（0.947）。摘要中的“失败率 55.2% → 6.9%”与此对应。
- **世界系 HMR（表 2）：** EMDB 上经 RL 精修后 WA-MPJPE100 70.60 mm、W-MPJPE100 175.93 mm、Jitter 8.14，优于 GVHMR/TRAM/WHAM/VideoMimic。
- **接触消融（表 3、图 5）：** PROX 上有无接触成功率均为 90.9%（多为坐姿，缺座面时动作退化为深蹲），但 Chamfer 与定性稳定性改善。
- **运行时间（附录 E）：** 300 帧 1440×1920 视频、单卡 RTX A6000 总计约 921 s，Visual SLAM 占 56%，平面拟合约 75 s；VideoMimic 同机准备阶段 1283 s。
- **场景感知策略（附录 F）：** 可选 PointNet 读取附近约 2.8k 点；主实验刻意用场景盲策略，避免策略补偿重建误差而掩盖几何问题。
- **局限（附录 C、D）：** 平面对强曲面/有机形状会呈面片化；假设静态刚性场景，无流体、可变形物、动态物体；RL 只做移动，不含 loco-manipulation；接触补全依赖 HMR 与接触预测精度，人体漂移会令补全平面倾斜或偏移；RL 不修改重建原语，仅把它们当作碰撞几何；平面间可能留缝。
- **与 sim2real 的接口（附录 G）：** 可直接替换 VideoMimic 四阶段管线的第 1、2 阶段。

## 对 wiki 的映射

- [`wiki/methods/crisp-real2sim.md`](../../wiki/methods/crisp-real2sim.md)
