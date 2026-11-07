# Latent RL Priors：面向数据高效灵巧操作的潜在强化学习先验

- **标题：** Latent RL Priors for Data-Efficient Dexterous Manipulation
- **论文：** <https://arxiv.org/abs/2611.08458>
- **全文版本：** <https://arxiv.org/html/2611.08458v1>，2026-11-05 v1。
- **作者：** Yixiao Wang、Peiyi Chen、Mustafa Mukadam、Shreekant Gayaka、Masayoshi Tomizuka、Son Tran。
- **机构：** 加利福尼亚大学伯克利分校、亚马逊。
- **项目页：** <https://yixiaowang7.github.io/latent-rl-prior-page/>
- **通读：** 已完成
- **核对范围：** 2026-11-07 通读 v1 HTML 全文 §1–6、表与图注、AI/Ethics/Reproducibility 声明及参考文献；该版本未列附录。项目页同步核查 Method、Setup、Results、Analysis、Limitations、Resources；不从图中估读正文未报告的成功率。

## 核心摘录

1. §3.2：从 SimToolReal 教师 checkpoint 微调程序化物体位姿任务，随机形状/尺寸/密度，收集 100000 成功轨迹；特权状态 tokenizer + Q-Former 输出 context，transformer action decoder 用 flow matching 学动作块。本体历史与 flow time 经 AdaLN，context 经 cross-attention，便于替换感知入口。
2. §3.3：视觉适配使用 DINOv2 + visual Q-Former 直接输出 decoder context，冻结 decoder，用动作损失训练视觉路径；视觉示范不需特权标签、latent target 或和 RL 数据配对。explicit transfer 是经过预训练状态编码器的对照，不是主方法。
3. 双臂：左右查询在一个 visual Q-Former 自注意力中协调，每臂 context 与各自本体历史调用同一个冻结单臂 decoder；不代表跨机器人形态迁移。
4. §4.1：MuJoCo Warp、KUKA iiwa + SHARPA；视觉数据为位姿对齐 4000、抬门 500、双臂装配 5000 episodes，来自各任务单独训练的 SAPG；不是人类实机示范。对齐判据为四关键点最大误差 ≤1.5/5 cm，10 个不必连续的 policy steps 达标。
5. §4.1–4.2：本方法训练 100k steps，π0.5 与 GR00T N1.7 全参数微调 30k；每 5k 存 checkpoint，最后五个各 256 rollout，报告最佳 checkpoint、通常三训练种子均值与标准差。抬门 50 对 500 示范、双臂 1000 对 5000 可超过全数据基线；节约的是下游视觉示范，非总训练计算。
6. §4.3–4.5：直接 latent interface 优于 explicit，保守视觉更新与冻结 decoder 更优；5 kg 抬门成功率 32.6% 对 π0.5 12.9%，示范量分别 50、500。线性 ridge 探针更易读出接触力，不等于直接输入力传感器或证明因果机制。
7. §5：全部仿真，同一臂手形态，需强特权教师与大量仿真数据；冻结先验可能继承教师失误并限制新控制行为；代码/数据承诺未来发布。

## 对 wiki 的映射

- [Latent RL Priors](../../wiki/entities/paper-latent-rl-priors.md)
- [项目页核查](../sites/latent-rl-priors.md)
